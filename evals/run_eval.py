"""A minimal evaluation harness.

Deliberately mechanical. Every metric here is computed by code, not judged by a
model, so the numbers mean the same thing every run and can't drift with the
grader's mood. That makes them boring and trustworthy -- which is what you want
from the thing you use to decide whether a prompt change helped.

    python evals/run_eval.py                  # run the default question set
    python evals/run_eval.py --questions x.json --out results.json

What it measures:
  citation_validity  fraction of citation markers backed by a retrieved source
                     (this should be 1.0 -- anything less is a pipeline bug)
  coverage           fraction of sub-questions that reached "answered"
  cited_fraction     fraction of retrieved sources the report actually used
  keyword_recall     fraction of expected keywords present in the report
  rounds / searches  what the run cost

`keyword_recall` is the only content signal and it is a crude one. Adding an
LLM judge -- with a model strictly stronger than the one being evaluated -- is
the obvious next step, and belongs behind a flag so the mechanical numbers stay
comparable across runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import citations  # noqa: E402
from graph.research_graph import compile_graph  # noqa: E402
from state import new_state  # noqa: E402

DEFAULT_QUESTIONS = Path(__file__).parent / "questions.json"


def score_run(question: dict, final: dict) -> dict:
    report = final.get("report", "")
    sources = final.get("sources", [])
    sub_questions = final.get("sub_questions", [])

    # Citation validity is measured before renumbering strips internal ids, so
    # we use the recorded issues rather than re-scanning the finished report.
    issues = final.get("citation_issues", [])
    total_markers = sum(
        len(citations.find_citations(sq.get("findings", ""))) for sq in sub_questions
    )
    validity = 1.0 if not total_markers else max(
        0.0, 1.0 - len(issues) / max(total_markers, 1)
    )

    answered = [sq for sq in sub_questions if sq.get("status") == "answered"]
    expected = [k.lower() for k in question.get("expect_keywords", [])]
    lowered = report.lower()
    recall = (
        sum(1 for k in expected if k in lowered) / len(expected) if expected else None
    )

    stats = citations.coverage(report, sources)

    return {
        "question": question["question"],
        "citation_validity": round(validity, 3),
        "coverage": round(len(answered) / len(sub_questions), 3) if sub_questions else 0.0,
        "cited_fraction": stats["cited_fraction"],
        "keyword_recall": round(recall, 3) if recall is not None else None,
        "sources_retrieved": len(sources),
        "sub_questions": len(sub_questions),
        "rounds": final.get("round", 0),
        "searches": final.get("search_count", 0),
        "stop_reason": final.get("stop_reason", ""),
        "citation_issues": len(issues),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    app = compile_graph()
    rows = []

    for i, question in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {question['question']}", flush=True)
        started = time.time()
        config = {"configurable": {"thread_id": f"eval-{i}-{int(started)}"}}
        try:
            app.invoke(new_state(question["question"]), config=config)
            final = app.get_state(config).values
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}")
            rows.append({"question": question["question"], "error": str(exc)})
            continue

        row = score_run(question, final)
        row["seconds"] = round(time.time() - started, 1)
        rows.append(row)
        print(
            f"    validity={row['citation_validity']} "
            f"coverage={row['coverage']} "
            f"recall={row['keyword_recall']} "
            f"{row['seconds']}s"
        )

    ok = [r for r in rows if "error" not in r]
    print("\n" + "=" * 60)
    print(f"{len(ok)}/{len(rows)} completed")
    if ok:
        for metric in ("citation_validity", "coverage", "cited_fraction", "seconds"):
            values = [r[metric] for r in ok if r.get(metric) is not None]
            if values:
                print(f"  {metric:<20} mean {statistics.mean(values):.3f}")
        recalls = [r["keyword_recall"] for r in ok if r.get("keyword_recall") is not None]
        if recalls:
            print(f"  {'keyword_recall':<20} mean {statistics.mean(recalls):.3f}")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
