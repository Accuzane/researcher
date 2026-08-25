"""Citation handling is the correctness core of this system, so it gets the
most tests. Every one of these is a claim about behaviour a model cannot be
trusted to provide."""

from __future__ import annotations

import citations
from tests.conftest import make_sources


SOURCES = make_sources(
    ("https://a.example/one", "Source A", "text a"),
    ("https://b.example/two", "Source B", "text b"),
    ("https://c.example/three", "Source C", "text c"),
)
A, B, C = (s["id"] for s in SOURCES)


class TestRemapLocal:
    def test_rewrites_local_numbers_to_global_ids(self):
        text, issues = citations.remap_local_citations("Fact one [1]. Fact two [2].", SOURCES)
        assert text == f"Fact one [{A}]. Fact two [{B}]."
        assert issues == []

    def test_handles_grouped_citations(self):
        text, _ = citations.remap_local_citations("Both agree [1, 3].", SOURCES)
        assert text == f"Both agree [{A}][{C}]."

    def test_out_of_range_is_dropped_and_reported(self):
        text, issues = citations.remap_local_citations("Claim [9].", SOURCES)
        assert text == "Claim ."
        assert len(issues) == 1
        assert "only 3" in issues[0]

    def test_leaves_markdown_links_alone(self):
        original = "See [the docs](https://x.example) for more."
        text, issues = citations.remap_local_citations(original, SOURCES)
        assert text == original
        assert issues == []


class TestValidate:
    def test_accepts_registry_backed_citations(self):
        assert citations.validate(f"Claim [{A}].", SOURCES) == []

    def test_flags_fabricated_citations(self):
        issues = citations.validate("Claim [Sdeadbeef].", SOURCES)
        assert len(issues) == 1
        assert "Sdeadbeef" in issues[0]

    def test_strip_unknown_removes_only_the_fabricated_one(self):
        text = citations.strip_unknown(f"Real [{A}] fake [Sdeadbeef].", SOURCES)
        assert text == f"Real [{A}] fake ."


class TestRenumber:
    def test_numbers_in_order_of_first_appearance(self):
        text, biblio = citations.renumber(f"X [{C}] Y [{A}] Z [{C}].", SOURCES)
        assert text == "X [1] Y [2] Z [1]."
        assert [s["id"] for s in biblio] == [C, A]

    def test_bibliography_excludes_uncited_sources(self):
        _, biblio = citations.renumber(f"Only A [{A}].", SOURCES)
        assert len(biblio) == 1

    def test_formats_bibliography_as_numbered_links(self):
        _, biblio = citations.renumber(f"[{B}]", SOURCES)
        assert citations.format_bibliography(biblio) == (
            "1. [Source B](https://b.example/two)"
        )

    def test_empty_bibliography_says_so(self):
        assert "No sources" in citations.format_bibliography([])


class TestCoverage:
    def test_counts_only_registry_backed_citations(self):
        stats = citations.coverage(f"[{A}] [{B}] [Sdeadbeef]", SOURCES)
        assert stats["sources_retrieved"] == 3
        assert stats["sources_cited"] == 2
        assert stats["cited_fraction"] == round(2 / 3, 3)

    def test_handles_no_sources(self):
        assert citations.coverage("nothing", [])["cited_fraction"] == 0.0
