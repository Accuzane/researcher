"""End-to-end runs of the research graph against fake models and fake search.

These are the tests that matter most. They exercise the parts a unit test can't
reach: `Send` fan-out into concurrent branches, reducers merging those branches
back together, the gap loop issuing follow-up queries and terminating, and the
citation pipeline surviving the whole trip from a sub-agent's `[1]` to the
report's `[1]` and bibliography.
"""

from __future__ import annotations

import re

from langgraph.checkpoint.memory import MemorySaver

import citations
from config import settings
from graph.research_graph import compile_graph
from state import new_state
from tests.conftest import FakeLLM


# --------------------------------------------------------------------------
# a scripted "well-behaved model"
# --------------------------------------------------------------------------

def make_handler(
    sub_questions: list[str],
    gaps_by_round: list[list[dict]] | None = None,
    report_prefix: str = "## Summary\n",
    fabricate_citation: bool = False,
    uncited_questions: tuple[str, ...] = (),
):
    """Build a handler that plays every role the graph asks of a model.

    `gaps_by_round` scripts what the gap analyser reports on each successive
    call, which is how the multi-round tests drive the loop.
    """
    state = {"gap_calls": 0}
    scripted_gaps = gaps_by_round or [[]]

    def handler(system: str, user: str, schema):
        name = schema.__name__ if schema else ""

        if name == "ResearchPlan":
            return {
                "topic": "Test Topic",
                "brief": "Establish the test facts.",
                "sub_questions": sub_questions,
            }

        if name == "SearchQueries":
            question = user.split("\n")[0].replace("Research question:", "").strip()
            return {"queries": [f"search for {question}"]}

        if name == "GapReport":
            index = min(state["gap_calls"], len(scripted_gaps) - 1)
            state["gap_calls"] += 1
            return {"gaps": scripted_gaps[index]}

        if name == "RouteDecision":
            return {"route": "research", "reason": "test"}

        # plain-text calls, distinguished by their system prompt
        if "You extract findings" in system:
            if any(q in user for q in uncited_questions):
                # Sources came back, but nothing in them supported an answer --
                # the model produces prose with no citations at all.
                return "The retrieved documents do not address this question."
            return (
                "The first established fact is significant and measurable [1]. "
                "A second corroborating fact appears in the same body of work [1]. "
                "A third independent observation supports the same conclusion "
                "across different conditions and timescales [1]."
            )

        if "You write research reports" in system:
            markers = re.findall(r"\[S[0-9a-f]{8}\]", user)
            body = report_prefix + " ".join(
                f"Finding number {i} is supported by the evidence. {marker}"
                for i, marker in enumerate(dict.fromkeys(markers), start=1)
            )
            if fabricate_citation:
                body += " An invented claim with no backing. [Sdeadbeef]"
            return body or report_prefix + "Nothing to report."

        return "unhandled"

    return handler


def run(app, query: str = "test query") -> dict:
    config = {"configurable": {"thread_id": "t1"}}
    app.invoke(new_state(query), config=config)
    return app.get_state(config).values


def build(handler, search_results, monkeypatch=None):
    import llm as llm_module
    from tools import search as search_module

    model = FakeLLM(handler)
    llm_module.set_models(model, model)

    calls: list[str] = []

    def backend(query: str, max_results: int):
        calls.append(query)
        return search_results(query)[:max_results]

    search_module.set_backend(backend)
    return compile_graph(checkpointer=MemorySaver()), calls


def three_docs(_query: str):
    return [
        {"url": "https://a.example/1", "title": "Doc A", "raw_content": "Content A " * 40},
        {"url": "https://b.example/2", "title": "Doc B", "raw_content": "Content B " * 40},
    ]


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------

class TestHappyPath:
    def test_produces_a_report_with_numbered_citations(self):
        app, _ = build(make_handler(["What is A?", "What is B?"]), three_docs)
        final = run(app)

        report = final["report"]
        assert report.startswith("# test query")
        assert "## Sources" in report
        assert re.search(r"\[\d+\]", report), "expected reader-facing numbered citations"
        assert not re.search(r"\[S[0-9a-f]{8}\]", report), "internal ids leaked into output"

    def test_bibliography_matches_the_markers_used(self):
        app, _ = build(make_handler(["What is A?", "What is B?"]), three_docs)
        report = run(app)["report"]

        used = {int(n) for n in re.findall(r"\[(\d+)\]", report.split("## Sources")[0])}
        listed = {int(n) for n in re.findall(r"^(\d+)\. \[", report.split("## Sources")[1], re.M)}
        assert used == listed

    def test_fans_out_one_branch_per_sub_question(self):
        app, _ = build(make_handler(["What is A?", "What is B?", "What is C?"]), three_docs)
        final = run(app)

        assert len(final["sub_questions"]) == 3
        assert all(sq["status"] == "answered" for sq in final["sub_questions"])

    def test_sources_are_deduplicated_across_branches(self):
        """Every branch retrieves the same two URLs; the registry must hold two."""
        app, _ = build(make_handler(["A?", "B?", "C?"]), three_docs)
        final = run(app)
        assert len(final["sources"]) == 2

    def test_search_count_accumulates_across_concurrent_branches(self):
        app, _ = build(make_handler(["A?", "B?", "C?"]), three_docs)
        final = run(app)
        assert final["search_count"] == 3  # one query per sub-question

    def test_stops_when_there_are_no_gaps(self):
        app, _ = build(make_handler(["A?"]), three_docs)
        assert run(app)["stop_reason"] == "all sub-questions answered"


# --------------------------------------------------------------------------
# the gap loop
# --------------------------------------------------------------------------

class TestGapLoop:
    def test_issues_the_follow_up_query_not_the_original(self):
        """The whole point of gap analysis: the second round searches for
        something different from the first."""
        handler = make_handler(
            ["What is A?"],
            gaps_by_round=[
                [{
                    "sub_question_id": "Q1",
                    "reason": "no figures given",
                    "follow_up_query": "specific numbers for A in 2026",
                }],
                [],
            ],
        )
        app, calls = build(handler, three_docs)
        run(app)

        assert "specific numbers for A in 2026" in calls
        assert len(calls) == 2, "expected exactly one follow-up round"

    def test_findings_accumulate_rather_than_being_overwritten(self):
        handler = make_handler(
            ["What is A?"],
            gaps_by_round=[
                [{"sub_question_id": "Q1", "reason": "thin", "follow_up_query": "more on A"}],
                [],
            ],
        )
        app, _ = build(handler, three_docs)
        final = run(app)
        assert "Additional findings:" in final["sub_questions"][0]["findings"]

    def test_round_budget_terminates_a_stubborn_loop(self):
        """A gap analyser that never clears anything must not loop forever."""
        never_satisfied = [[{
            "sub_question_id": "Q1",
            "reason": "still not enough",
            "follow_up_query": "try again",
        }]]
        app, calls = build(make_handler(["A?"], gaps_by_round=never_satisfied), three_docs)
        final = run(app)

        assert "round budget" in final["stop_reason"]
        assert final["round"] == settings.max_rounds
        assert len(calls) <= settings.max_rounds

    def test_search_budget_terminates_the_loop(self, monkeypatch):
        monkeypatch.setattr(settings, "max_searches", 1)
        monkeypatch.setattr(settings, "max_rounds", 99)
        never_satisfied = [[{
            "sub_question_id": "Q1", "reason": "x", "follow_up_query": "y",
        }]]
        app, _ = build(make_handler(["A?"], gaps_by_round=never_satisfied), three_docs)
        assert "search budget" in run(app)["stop_reason"]

    def test_a_gap_for_an_unknown_sub_question_is_ignored(self):
        bogus = [[{
            "sub_question_id": "Q99", "reason": "hallucinated id", "follow_up_query": "x",
        }]]
        app, calls = build(make_handler(["A?"], gaps_by_round=bogus), three_docs)
        final = run(app)
        assert final["report"]
        assert len(calls) == 1, "a bogus gap id must not trigger a research round"


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------

class TestFailureModes:
    def test_a_fabricated_citation_is_stripped_from_the_report(self):
        handler = make_handler(["A?"], fabricate_citation=True)
        app, _ = build(handler, three_docs)
        final = run(app)

        assert "Sdeadbeef" not in final["report"]
        assert any("Sdeadbeef" in issue for issue in final["citation_issues"])
        assert "dropped as unverifiable" in final["report"]

    def test_no_search_results_still_produces_output(self):
        app, _ = build(make_handler(["A?"]), lambda q: [])
        final = run(app)

        assert final["report"]
        assert "No sources could be retrieved" in final["report"]
        assert final["sub_questions"][0]["status"] == "unanswered"

    def test_a_failing_search_backend_does_not_crash_the_run(self):
        import llm as llm_module
        from tools import search as search_module

        model = FakeLLM(make_handler(["A?"]))
        llm_module.set_models(model, model)

        def exploding(query: str, max_results: int):
            raise RuntimeError("tavily rate limit")

        search_module.set_backend(exploding)
        app = compile_graph(checkpointer=MemorySaver())

        final = run(app)
        assert final["report"]

    def test_a_planner_failure_degrades_to_one_sub_question(self):
        def handler(system, user, schema):
            if schema and schema.__name__ == "ResearchPlan":
                raise RuntimeError("model unavailable")
            return make_handler(["unused"])(system, user, schema)

        app, _ = build(handler, three_docs)
        final = run(app)

        assert len(final["sub_questions"]) == 1
        assert final["sub_questions"][0]["question"] == "test query"
        assert final["report"]

    def test_unresolved_sub_questions_are_declared_in_the_report(self):
        """Sources were found, but one sub-question stayed unsupported. The
        report must say so rather than quietly presenting partial coverage as
        a complete answer."""
        handler = make_handler(
            ["What is A?", "What is B?"],
            gaps_by_round=[[]],  # the analyser clears it; the mechanical signal does not
            uncited_questions=("What is B?",),
        )
        app, _ = build(handler, three_docs)
        final = run(app)

        statuses = {sq["id"]: sq["status"] for sq in final["sub_questions"]}
        assert statuses["Q2"] == "unanswered"
        assert "unresolved" in final["report"].lower()
        assert "Q2" in final["report"]

    def test_no_sources_at_all_says_so_plainly(self):
        app, _ = build(make_handler(["A?"]), lambda q: [])
        report = run(app)["report"]
        assert "No sources could be retrieved" in report
        assert "TAVILY_API_KEY" in report


# --------------------------------------------------------------------------
# citation integrity across the whole pipeline
# --------------------------------------------------------------------------

class TestCitationIntegrity:
    def test_every_number_in_the_report_resolves_to_a_real_url(self):
        app, _ = build(make_handler(["A?", "B?"]), three_docs)
        final = run(app)
        report = final["report"]

        body, _, sources_block = report.partition("## Sources")
        urls = re.findall(r"\((https?://[^)]+)\)", sources_block)
        retrieved = {s["url"] for s in final["sources"]}

        assert urls, "bibliography should list at least one source"
        assert set(urls) <= retrieved, "report cites a URL that was never retrieved"

    def test_internal_ids_never_reach_the_reader(self):
        app, _ = build(make_handler(["A?"]), three_docs)
        final = run(app)
        assert citations.find_citations(final["report"]) == []
