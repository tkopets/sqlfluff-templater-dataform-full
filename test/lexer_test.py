"""Unit tests for the context-tracking block lexer (`parsing_method = lexer`).

These do not need the Dataform CLI: they exercise `_find_template_blocks_by_lexer`
and `_annotate_sqlx_with_markers` directly (both pure Python).

Three layers:
  1. Equivalence -- on every ordinary fixture the lexer must annotate identically
     to the previous default (`regex`), so switching the default is a safe no-op;
     the fixtures written to defeat `regex` must genuinely differ.
  2. Golden edge cases -- inputs where the `regex` finder mis-slices.
  3. Reference parity -- behaviours checked against Dataform itself; see
     REFERENCE_PARITY below.
"""

import logging
from pathlib import Path

import pytest
from sqlfluff.core import FluffConfig
from sqlfluff.core.errors import SQLFluffUserError

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
        for b in t._find_template_blocks_by_lexer(src)[0]
    ]


@pytest.mark.parametrize("fname", FIXTURES)
def test_fixtures_lex_cleanly(templater: DataformTemplaterFull, fname: str):
    """Real fixtures must lex balanced, so the lexer is the active path and does
    not silently fall back to regex everywhere."""
    _, balanced = templater._find_template_blocks_by_lexer(
        (FIXTURE_DIR / fname).read_text()
    )
    assert balanced


# Fixtures written specifically to defeat the `regex` finder. Every other
# fixture must annotate identically under both, so switching the default is a
# no-op for files that avoid these constructs.
DIVERGENT_FIXTURES = {
    "string_with_comment_marker.sqlx",
    "js_block_with_brace_in_string.sqlx",
}


@pytest.mark.parametrize("fname", [f for f in FIXTURES if f not in DIVERGENT_FIXTURES])
def test_lexer_annotates_like_regex_on_fixtures(
    templater: DataformTemplaterFull, fname: str
):
    """On ordinary fixtures the lexer must produce the same annotated file as regex."""
    src = (FIXTURE_DIR / fname).read_text()

    lexer_blocks = templater._find_template_blocks_by_lexer(src)[0]
    regex_blocks = templater._find_template_blocks_by_regex(src)

    lexer_out = templater._annotate_sqlx_with_markers(src, fname, lexer_blocks)
    regex_out = templater._annotate_sqlx_with_markers(src, fname, regex_blocks)

    assert lexer_out == regex_out


@pytest.mark.parametrize("fname", sorted(DIVERGENT_FIXTURES))
def test_divergent_fixtures_actually_defeat_regex(
    templater: DataformTemplaterFull, fname: str
):
    """...and these must really differ, or they are not testing anything.

    Both are also end-to-end fixtures: `templater_test.py` compiles them and
    compares against `expected_output/`, which only matches under the lexer.
    """
    src = (FIXTURE_DIR / fname).read_text()

    lexer_blocks = templater._find_template_blocks_by_lexer(src)[0]
    regex_blocks = templater._find_template_blocks_by_regex(src)

    assert lexer_blocks != regex_blocks


# (input, expected [(block_type, exact text)]) for cases the `regex` finder gets
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


def test_lexer_fixes_regex_brace_in_string(templater: DataformTemplaterFull):
    """The lexer keeps a whole `js {}` block that the regex finder truncates on a
    `}` inside a string -- demonstrating why it is the better default."""
    src = 'js { const x = "}"; }'

    lexer_out = _spans(templater, src)
    regex_blocks = templater._find_template_blocks_by_regex(src)
    regex_out = [(b.block_type, src[b.outer_start : b.outer_end]) for b in regex_blocks]

    assert lexer_out == [(BlockType.JS, src)]
    assert regex_out != lexer_out


COMMENT_TYPES = (BlockType.SQL_LINE_COMMENT, BlockType.SQL_BLOCK_COMMENT)


@pytest.mark.parametrize("fname", FIXTURES)
def test_lexer_blocks_are_ordered_and_non_overlapping(
    templater: DataformTemplaterFull, fname: str
):
    """`_annotate_sqlx_with_markers` walks blocks in order and assumes they never
    overlap, so hold the lexer to that contract on every fixture."""
    src = (FIXTURE_DIR / fname).read_text()
    blocks = templater._find_template_blocks_by_lexer(src)[0]

    prev_end = 0
    for b in blocks:
        assert b.outer_start >= prev_end
        if b.block_type in COMMENT_TYPES:
            # Comments carry no inner content: the span is the whole comment.
            assert b.outer_start == b.inner_start
            assert b.inner_end == b.outer_end
        else:
            assert b.outer_start < b.inner_start <= b.inner_end < b.outer_end
        assert src[b.outer_start : b.outer_end]
        prev_end = b.outer_end


def test_dispatch_falls_back_to_regex_when_lexer_unbalanced(
    templater: DataformTemplaterFull, caplog
):
    """An escaped backtick in a `${}` expression is unmodellable, so the lexer
    bails and drops the placeholder; dispatch must notice and fall back."""
    src = r"""config { type: "view" }
select ${ `x\`y`.length } as v"""

    # Raw lexer: unbalanced, and the placeholder is dropped.
    blocks, balanced = templater._find_template_blocks_by_lexer(src)
    assert balanced is False
    assert BlockType.TEMPLATED not in [b.block_type for b in blocks]

    # Dispatch (config=None -> defaults to lexer) notices and falls back to regex.
    with caplog.at_level(logging.WARNING, logger="sqlfluff.templater"):
        dispatched = templater._find_template_blocks(src, None, "model.sqlx")
    assert BlockType.TEMPLATED in [b.block_type for b in dispatched]
    assert "falling back to the regex parser" in caplog.text
    assert "model.sqlx" in caplog.text


# Expectations checked against Dataform itself. Several look like bugs and are
# not: don't "fix" them without re-checking what Dataform does with the input.
REFERENCE_PARITY = [
    # Openers need exactly one space before `{`.
    pytest.param(
        'config  { type: "view" }\nselect ${x}',
        [(BlockType.TEMPLATED, "${x}")],
        id="two-spaces-before-brace-is-not-a-block",
    ),
    pytest.param(
        'config\n{ type: "view" }\nselect ${x}',
        [(BlockType.TEMPLATED, "${x}")],
        id="newline-before-brace-is-not-a-block",
    ),
    # ...and no word-boundary guard.
    pytest.param(
        "select 1 from t\nwhere nojs { x }",
        [(BlockType.JS, "js { x }")],
        id="keyword-suffix-opens-a-block",
    ),
    # `input` takes a restricted charset and optional extra names.
    pytest.param(
        'input "a", "b" {\n  select 1\n}\nselect ${x}',
        [
            (BlockType.INPUT, 'input "a", "b" {\n  select 1\n}'),
            (BlockType.TEMPLATED, "${x}"),
        ],
        id="input-block-with-multiple-names",
    ),
    pytest.param(
        'input "a.b" {\n  select 1\n}\nselect ${x}',
        [(BlockType.TEMPLATED, "${x}")],
        id="input-name-outside-charset-is-not-a-block",
    ),
    # An unterminated `/*` is ordinary text, so it must not swallow the file.
    pytest.param(
        "select 1 /* oops\nselect ${x}",
        [(BlockType.TEMPLATED, "${x}")],
        id="unterminated-block-comment-is-not-a-comment",
    ),
    pytest.param(
        "js { /* unterminated } more ${x}",
        [(BlockType.JS, "js { /* unterminated }"), (BlockType.TEMPLATED, "${x}")],
        id="unterminated-js-block-comment-does-not-eat-closing-brace",
    ),
    # A bare `{` does not nest in an inner-SQL block; the regex finder disagrees.
    pytest.param(
        "pre_operations {\n  select struct { }\n}\nselect ${x}",
        [
            (BlockType.PRE_OPERATIONS, "pre_operations {\n  select struct { }"),
            (BlockType.TEMPLATED, "${x}"),
        ],
        id="bare-brace-does-not-nest-in-inner-sql",
    ),
    # A JS regex literal is not modelled either, so `}` inside one closes.
    pytest.param(
        "js { const re = /}/; }",
        [(BlockType.JS, "js { const re = /}")],
        id="js-regex-literal-is-not-modelled",
    ),
    # `#` is not a comment marker to Dataform, so the placeholder is templated.
    pytest.param(
        "# c ${x}\nselect 1",
        [(BlockType.TEMPLATED, "${x}")],
        id="hash-is-not-a-comment",
    ),
    # All string flavours interpolate; comments do not.
    pytest.param(
        "select '''${a}''' as x",
        [(BlockType.TEMPLATED, "${a}")],
        id="placeholder-in-triple-single-quoted-string",
    ),
    pytest.param(
        'select """${a}""" as x',
        [(BlockType.TEMPLATED, "${a}")],
        id="placeholder-in-triple-double-quoted-string",
    ),
    pytest.param(
        "/* ${a} */\nselect ${b}",
        [
            (BlockType.SQL_BLOCK_COMMENT, "/* ${a} */"),
            (BlockType.TEMPLATED, "${b}"),
        ],
        id="placeholder-in-block-comment-is-ignored",
    ),
    pytest.param(
        "pre_operations {\n  -- ${a}\n  select 1\n}\nselect ${b}",
        [
            (BlockType.PRE_OPERATIONS, "pre_operations {\n  -- ${a}\n  select 1\n}"),
            (BlockType.TEMPLATED, "${b}"),
        ],
        id="comment-inside-inner-sql-block",
    ),
]


@pytest.mark.parametrize(("src", "expected"), REFERENCE_PARITY)
def test_lexer_matches_dataform_reference(
    templater: DataformTemplaterFull,
    src: str,
    expected: list[tuple[BlockType, str]],
):
    assert _spans(templater, src) == expected


def test_placeholder_in_double_quoted_string_diverges_from_reference(
    templater: DataformTemplaterFull,
):
    """Dataform's lexer misses this placeholder, but its compiler still
    interpolates it (the string escaper rewrites only `\\` and backticks), so we
    have to mark it to map the compiled SQL back."""
    assert _spans(templater, 'select "${b}" as x') == [(BlockType.TEMPLATED, "${b}")]


def test_statement_separator_is_not_templated(templater: DataformTemplaterFull):
    """Dataform spans `---` plus its surrounding whitespace; we start at the
    first `-`. Neither is templated and comments emit no slices, so the
    annotated output is byte-identical either way."""
    src = "select 1\n  ---  \nselect 2"

    assert _spans(templater, src) == [(BlockType.SQL_LINE_COMMENT, "---  ")]
    blocks = templater._find_template_blocks_by_lexer(src)[0]
    assert templater._annotate_sqlx_with_markers(src, "m.sqlx", blocks) == src


@pytest.mark.parametrize("parsing_method", ["lexer", "regex"])
def test_dispatch_honours_configured_parsing_method(
    templater: DataformTemplaterFull, parsing_method: str
):
    """Each documented method is reachable through a real FluffConfig."""
    config = FluffConfig(
        configs={
            "core": {"templater": "dataform-full", "dialect": "bigquery"},
            "templater": {"dataform-full": {"parsing_method": parsing_method}},
        }
    )
    blocks = templater._find_template_blocks("select ${x}", config)

    assert [b.block_type for b in blocks] == [BlockType.TEMPLATED]


def test_dispatch_defaults_to_lexer_without_config(templater: DataformTemplaterFull):
    """No config at all must still use the lexer, not a legacy method."""
    src = 'js { const x = "}"; }'

    assert (
        templater._find_template_blocks(src, None)
        == templater._find_template_blocks_by_lexer(src)[0]
    )


def test_dispatch_rejects_unknown_parsing_method(templater: DataformTemplaterFull):
    config = FluffConfig(
        configs={
            "core": {"templater": "dataform-full", "dialect": "bigquery"},
            "templater": {"dataform-full": {"parsing_method": "nonsense"}},
        }
    )
    with pytest.raises(SQLFluffUserError) as excinfo:
        templater._find_template_blocks("select 1", config)

    assert "nonsense" in str(excinfo.value)
    assert "'lexer' or 'regex'" in str(excinfo.value)
