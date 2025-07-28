"""Defines the hook endpoints for the Dataform templater plugin."""

from sqlfluff.core.plugin import hookimpl

from sqlfluff_templater_dataform_full.templater import DataformTemplaterFull


@hookimpl
def get_templaters():
    """Get templaters."""
    return [DataformTemplaterFull]
