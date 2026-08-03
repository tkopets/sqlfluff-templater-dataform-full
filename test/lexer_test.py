"""Unit tests for the context-tracking block lexer (`parsing_method = lexer`).

These do not need the Dataform CLI: they exercise `_find_template_blocks_by_lexer`
and `_annotate_sqlx_with_markers` directly (both pure Python).

Two layers:
  1. Equivalence -- on every input fixture the lexer must annotate identically to
     the previous default (`regex`), so switching the default is a safe no-op for
     real files.
  2. Golden edge cases -- inputs where `regex`/`char` mis-slice (braces, `${`,
     comment markers inside strings/comments/template literals) and the lexer is
     correct.
"""

import logging
from pathlib import Path

import pytest

from sqlfluff_templater_dataform_full.templater import (
    BlockType,
    DataformTemplaterFull,
)

FIXTURE_DIR = Path("test/fixtures/dataform/dataform_project/definitions/input")
FIXTURES = sorted(p.name for p in FIXTURE_DIR.glob("*.sqlx"))


@pytest.fixture
def templater() -> DataformTemplaterFull:
    return DataformTemplaterFull()


def _spans(t: DataformTemplaterFull, src: str) -> list[tuple[BlockType, str]]:
    """(block_type, exact source text) for each block the lexer finds."""
    return [
        (b.block_type, src[b.outer_start : b.outer_end])
        for b in t._find_template_blocks_by_lexer(src)
    ]


@pytest.mark.parametrize("fname", FIXTURES)
def test_fixtures_lex_cleanly(templater: DataformTemplaterFull, fname: str):
    """Real fixtures must lex balanced, so the lexer is the active path and does
    not silently fall back to regex everywhere."""
    _, balanced = templater._lex((FIXTURE_DIR / fname).read_text())
    assert balanced


@pytest.mark.parametrize("fname", FIXTURES)
def test_lexer_annotates_like_regex_on_fixtures(
    templater: DataformTemplaterFull, fname: str
):
    """On real fixtures the lexer must produce the same annotated file as regex."""
    src = (FIXTURE_DIR / fname).read_text()

    lexer_blocks = templater._find_template_blocks_by_lexer(src)
    regex_blocks = templater._find_template_blocks_by_regex(src)

    lexer_out = templater._annotate_sqlx_with_markers(src, fname, lexer_blocks)
    regex_out = templater._annotate_sqlx_with_markers(src, fname, regex_blocks)

    assert lexer_out == regex_out


# (input, expected [(block_type, exact text)]) for cases the naive finders get
# wrong. The lexer tracks string/comment/template context, so these are correct.
GOLDEN = [
    pytest.param(
        'js { const x = "}"; }',
        [(BlockType.JS, 'js { const x = "}"; }')],
        id="brace-inside-js-string",
    ),
    pytest.param(
        "js {\n  // }\n  foo();\n}",
        [(BlockType.JS, "js {\n  // }\n  foo();\n}")],
        id="brace-inside-js-line-comment",
    ),
    pytest.param(
        "config { d: `a } b { c` }",
        [(BlockType.CONFIG, "config { d: `a } b { c` }")],
        id="braces-inside-config-template-literal",
    ),
    pytest.param(
        "select '${a}', \"${b}\"",
        [(BlockType.TEMPLATED, "${a}"), (BlockType.TEMPLATED, "${b}")],
        id="placeholder-in-single-and-double-quoted-strings",
    ),
    pytest.param(
        "select 'a -- b ${c} d'",
        [(BlockType.TEMPLATED, "${c}")],
        id="dashes-in-string-are-not-a-comment",
    ),
    pytest.param(
        "-- c ${x}\nselect ${y}",
        [
            (BlockType.SQL_LINE_COMMENT, "-- c ${x}"),
            (BlockType.TEMPLATED, "${y}"),
        ],
        id="placeholder-in-comment-is-ignored",
    ),
    pytest.param(
        '${ a("}") + b }',
        [(BlockType.TEMPLATED, '${ a("}") + b }')],
        id="brace-inside-templated-expression-string",
    ),
    pytest.param(
        "pre_operations {\n  select ${ref('t')} , '}'\n}\nselect ${x}",
        [
            (
                BlockType.PRE_OPERATIONS,
                "pre_operations {\n  select ${ref('t')} , '}'\n}",
            ),
            (BlockType.TEMPLATED, "${x}"),
        ],
        id="pre-operations-opaque-with-brace-in-string",
    ),
]


@pytest.mark.parametrize(("src", "expected"), GOLDEN)
def test_lexer_golden_edge_cases(
    templater: DataformTemplaterFull,
    src: str,
    expected: list[tuple[BlockType, str]],
):
    assert _spans(templater, src) == expected


def test_lexer_fixes_char_brace_in_string(templater: DataformTemplaterFull):
    """The lexer keeps a whole `js {}` block that the char finder truncates on a
    `}` inside a string -- demonstrating why it is the better default."""
    src = 'js { const x = "}"; }'

    lexer_out = _spans(templater, src)
    char_blocks = templater._find_template_blocks_by_char(src)
    char_out = [(b.block_type, src[b.outer_start : b.outer_end]) for b in char_blocks]

    assert lexer_out == [(BlockType.JS, src)]
    assert char_out != lexer_out


def test_lexer_blocks_are_ordered_and_non_overlapping(
    templater: DataformTemplaterFull,
):
    src = (FIXTURE_DIR / "config_js_query.sqlx").read_text()
    blocks = templater._find_template_blocks_by_lexer(src)

    prev_end = -1
    for b in blocks:
        assert b.outer_start >= prev_end
        assert b.outer_start < b.inner_start <= b.inner_end < b.outer_end
        prev_end = b.outer_end


def test_dispatch_falls_back_to_regex_when_lexer_unbalanced(
    templater: DataformTemplaterFull, caplog
):
    """An escaped backtick inside a `${}` expression is a construct the lexer
    can't model (Dataform compiles it to `select 3 as v`). The raw lexer bails
    (unbalanced, dropping the placeholder), so the dispatch must notice and fall
    back to the regex parser, which recovers the placeholder."""
    src = r"""config { type: "view" }
select ${ `x\`y`.length } as v"""

    # Raw lexer: unbalanced, and the placeholder is dropped.
    blocks, balanced = templater._lex(src)
    assert balanced is False
    assert BlockType.TEMPLATED not in [b.block_type for b in blocks]

    # Dispatch (config=None -> defaults to lexer) notices and falls back to regex.
    with caplog.at_level(logging.WARNING, logger="sqlfluff.templater"):
        dispatched = templater._find_template_blocks(src, None, "model.sqlx")
    assert BlockType.TEMPLATED in [b.block_type for b in dispatched]
    assert "falling back to the regex parser" in caplog.text
    assert "model.sqlx" in caplog.text
