"""Tests for the dataform templater."""

from pathlib import Path

import pytest
from sqlfluff.core import FluffConfig, Lexer
from sqlfluff.core.templaters import RawTemplater
from sqlfluff.core.types import ConfigMappingType


def _run_templater_and_verify_result(
    dataform_templater: RawTemplater,
    project_dir: Path,
    fname: str,
    dataform_fluff_config: ConfigMappingType,
    assets_temp_dir: Path,
):
    input_path = Path(project_dir) / "definitions/input" / fname

    expected_out_file = fname.removesuffix(".sqlx") + ".sql"
    expected_out_path = Path(assets_temp_dir) / "expected_output" / expected_out_file

    config = FluffConfig(configs=dataform_fluff_config)
    templated_file, _ = dataform_templater.process(
        in_str=input_path.read_text(),
        fname=str(input_path),
        config=config,
    )

    assert str(templated_file) == expected_out_path.read_text()
    lexer = Lexer(config=config)
    _, lexing_violations = lexer.lex(templated_file)
    assert not lexing_violations


@pytest.mark.parametrize(
    "fname",
    [
        "config_js_query.sqlx",
        "config_query.sqlx",
        "js_block_sandwich.sqlx",
        "no_template.sqlx",
        "sql_line_comment_ignored.sqlx",
        "sql_block_comment_ignored.sqlx",
    ],
)
def test__templater_dataform_templating_result(
    project_dir: Path,
    dataform_templater: RawTemplater,
    fname: str,
    dataform_fluff_config: ConfigMappingType,
    assets_temp_dir: Path,
):
    """Test that input sql file gets templated into output sql file."""
    _run_templater_and_verify_result(
        dataform_templater=dataform_templater,
        project_dir=project_dir,
        fname=fname,
        dataform_fluff_config=dataform_fluff_config,
        assets_temp_dir=assets_temp_dir,
    )
