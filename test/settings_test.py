import shutil
from pathlib import Path

import pytest
from sqlfluff.core import FluffConfig
from sqlfluff.core.errors import SQLFluffUserError
from sqlfluff.core.types import ConfigMappingType


def test_dataform_executable_config(
    project_dir: Path,
    dataform_fluff_config: ConfigMappingType,
    assets_temp_dir: Path,
    monkeypatch,
):
    """Test that the dataform_executable config option is respected."""
    # Set a fake dataform executable path
    monkeypatch.setenv("DATAFORM_EXECUTABLE", "/path/to/nonexistent/dataform")

    # Get a fresh templater
    templater = FluffConfig(configs=dataform_fluff_config).get_templater()

    # Should fail because the executable doesn't exist
    with pytest.raises(SQLFluffUserError) as excinfo:
        _ = templater.process(
            in_str=Path(project_dir, "definitions/input/no_template.sqlx").read_text(),
            fname=str(Path(project_dir, "definitions/input/no_template.sqlx")),
            config=FluffConfig(configs=dataform_fluff_config),
        )
    assert "Failed to compile Dataform project" in str(excinfo.value)

    # Now, provide the correct path via config
    if isinstance(dataform_fluff_config["templater"], dict) and isinstance(
        dataform_fluff_config["templater"]["dataform-full"], dict
    ):
        dataform_fluff_config["templater"]["dataform-full"]["dataform_executable"] = (
            shutil.which("dataform")
        )

    # Get a fresh templater to pick up the new config
    templater = FluffConfig(configs=dataform_fluff_config).get_templater()

    # Should succeed
    templated_file, _ = templater.process(
        in_str=Path(project_dir, "definitions/input/no_template.sqlx").read_text(),
        fname=str(Path(project_dir, "definitions/input/no_template.sqlx")),
        config=FluffConfig(configs=dataform_fluff_config),
    )
    expected_out_path = assets_temp_dir / "expected_output" / "no_template.sql"
    assert str(templated_file) == expected_out_path.read_text()
