"""pytest fixtures."""

import shutil
from pathlib import Path

import pytest
from sqlfluff.core import FluffConfig
from sqlfluff.core.templaters import RawTemplater


def pytest_report_header() -> list[str]:
    """Return a list of strings to be displayed in the header of the report."""
    return []


@pytest.fixture(scope="session")
def assets_temp_dir():
    """Fixture for a temporary fixture assets directory."""
    src = Path("test/fixtures/dataform")
    tmp = Path("test/temp_dataform_project")
    tmp.mkdir(exist_ok=True)
    shutil.copytree(src, tmp, dirs_exist_ok=True)

    yield tmp

    shutil.rmtree(tmp)


@pytest.fixture
def dataform_fluff_config(assets_temp_dir: Path):
    """Returns SQLFluff dataform configuration dictionary."""
    return {
        "core": {
            "templater": "dataform-full",
            "dialect": "bigquery",
        },
        "templater": {
            "dataform-full": {
                "project_dir": f"{assets_temp_dir}/dataform_project",
            },
        },
    }


@pytest.fixture
def project_dir(dataform_fluff_config) -> Path:
    """Returns the dataform project directory."""
    return Path(dataform_fluff_config["templater"]["dataform-full"]["project_dir"])


@pytest.fixture
def dataform_templater() -> RawTemplater:
    """Returns an instance of the templater."""
    return FluffConfig(
        overrides={"dialect": "bigquery", "templater": "dataform-full"}
    ).get_templater()
