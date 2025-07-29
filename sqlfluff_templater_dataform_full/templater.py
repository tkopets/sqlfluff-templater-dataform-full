import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypedDict, cast, final

from sqlfluff.core import FluffConfig
from sqlfluff.core.errors import SQLFluffUserError, SQLTemplaterError
from sqlfluff.core.formatter import FormatterInterface
from sqlfluff.core.helpers.slice import zero_slice
from sqlfluff.core.templaters.base import (
    RawFileSlice,
    RawTemplater,
    TemplatedFile,
    TemplatedFileSlice,
    large_file_check,
)
from typing_extensions import NotRequired, override

templater_logger = logging.getLogger("sqlfluff.templater")

MARKER_ID_FORMAT = "_SQLFLUFF_TPL_{}_"
MARKER_RE_PAIR = re.compile(
    r"/\*\~START_MARKER:(_SQLFLUFF_TPL_\d+_)~\*/(?P<templated_content>.*?)/\*\~END_MARKER:\1~\*/",
    re.DOTALL,
)


class DataformCompiledTable(TypedDict):
    """Represents a single compiled table object from Dataform."""

    fileName: str
    query: str


DataformCompiledAssertion = DataformCompiledTable


class DataformCompiledOperation(TypedDict):
    """Represents a single compiled object (table, assertion, etc.) from Dataform."""

    fileName: str
    queries: list[str]


class DataformCompilationResult(TypedDict):
    """Represents the top-level JSON output of `dataform compile`."""

    tables: NotRequired[list[DataformCompiledTable]]
    assertions: NotRequired[list[DataformCompiledAssertion]]
    operations: NotRequired[list[DataformCompiledOperation]]


class BlockType(str, Enum):
    CONFIG = "config"
    JS = "js"
    PRE_OPERATIONS = "pre_operations"
    POST_OPERATIONS = "post_operations"
    TEMPLATED = "templated"
    SQL_LINE_COMMENT = "comment_line"
    SQL_BLOCK_COMMENT = "comment_block"


@dataclass
class BlockSpan:
    outer_start: int
    inner_start: int
    inner_end: int
    outer_end: int
    block_type: BlockType


@dataclass
class CompilationCache:
    """Caches compilation artifacts for a file."""

    blocks: list[BlockSpan]
    compiled_sql: str


@final
class DataformTemplaterFull(RawTemplater):
    """
    A templater for Dataform .sqlx files.

    This templater works by:
    1. Finding all `${...}` template blocks in the source file.
    2. Wrapping the content of these blocks in an IIFE (Immediately Invoked
       Function Expression) that also embeds unique marker comments.
    3. Creating a temporary Dataform project and running `dataform compile`.
    4. Parsing the compiled SQL to find the markers.
    5. Using the markers to construct the slices for SQLFluff, mapping the
       compiled output back to the original source.
    """

    name = "dataform-full"

    def __init__(self, override_context: Optional[dict[str, Any]] = None):
        self.working_dir = os.getcwd()
        self._sequenced_files: list[str] = []
        self._compilation_cache: dict[str, CompilationCache] = {}
        super().__init__(override_context=override_context)

    def _get_dataform_executable(self, config: Optional[FluffConfig]) -> str:
        """Get the dataform executable path from the configuration."""
        config_section_val = None
        if config is not None:
            config_section_val = config.get_section(  # pyright:ignore[reportAny]
                (self.templater_selector, self.name, "dataform_executable")
            )
        dataform_executable = (
            str(config_section_val)  # pyright:ignore[reportAny]
            if config_section_val is not None
            else os.getenv("DATAFORM_EXECUTABLE") or "dataform"
        )
        templater_logger.debug(f"Using dataform executable: {dataform_executable}")

        return dataform_executable

    def _get_project_dir(self, config: Optional[FluffConfig]) -> str:
        """Get Dataform project directory from the configuration.

        Defaults to the working directory.
        """
        config_section_val = None
        if config is not None:
            config_section_val = config.get_section(  # pyright:ignore[reportAny]
                (self.templater_selector, self.name, "project_dir")
            )
        config_project_dir: Optional[str] = (
            str(config_section_val)  # pyright:ignore[reportAny]
            if config_section_val is not None
            else None
        )

        project_dir = os.path.abspath(
            os.path.expanduser(
                config_project_dir or os.getenv("DATAFORM_PROJECT_DIR") or os.getcwd()
            )
        )
        if not os.path.exists(project_dir):
            templater_logger.error(f"project_dir: {project_dir} does not exists.")
        templater_logger.debug(f"Using project_dir: {project_dir}")

        return project_dir

    def _find_template_blocks_by_regex(self, raw_str: str) -> list[BlockSpan]:
        """
        Uses a regex-based parser to find all `${...}`, `js {...}`,
        and `config {...}` blocks.
        """
        # regex to match all kinds of blocks and comments
        # - config, js, pre_operations, post_operations blocks (handles nested braces)
        # - ${...} templated blocks (handles nested braces)
        # - SQL line and block comments
        block_regex = re.compile(
            r"(?P<block_type_outer>config|js|pre_operations|post_operations)\s*\{(?P<block_content>(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}|"
            r"(?P<templated_block>\$\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})|"
            r"(?P<comment_block>/\*.*?\*/)|"
            r"(?P<comment_line>--.*?(?=\n|$))",
            re.DOTALL,
        )

        results: list[BlockSpan] = []
        for match in block_regex.finditer(raw_str):
            outer_start, outer_end = match.span()

            if match.group("block_type_outer"):
                block_type = BlockType(match.group("block_type_outer"))
                inner_start = match.start("block_content")
                inner_end = match.end("block_content")
                results.append(
                    BlockSpan(
                        outer_start,
                        inner_start,
                        inner_end,
                        outer_end,
                        block_type,
                    )
                )
            elif match.group("templated_block"):
                # for ${...}, inner content is between ${ and }
                inner_start = outer_start + 2
                inner_end = outer_end - 1
                results.append(
                    BlockSpan(
                        outer_start,
                        inner_start,
                        inner_end,
                        outer_end,
                        BlockType.TEMPLATED,
                    )
                )
            elif match.group("comment_block"):
                results.append(
                    BlockSpan(
                        outer_start,
                        outer_start,
                        outer_end,
                        outer_end,
                        BlockType.SQL_BLOCK_COMMENT,
                    )
                )
            elif match.group("comment_line"):
                results.append(
                    BlockSpan(
                        outer_start,
                        outer_start,
                        outer_end,
                        outer_end,
                        BlockType.SQL_LINE_COMMENT,
                    )
                )
        return results

    def _find_template_blocks(
        self, raw_str: str, config: Optional[FluffConfig]
    ) -> list[BlockSpan]:
        """
        Finds all template blocks in the raw string, using the configured parsing method
        """
        parsing_method = "regex"
        if config:
            parsing_method = (
                config.get_section(
                    (self.templater_selector, self.name, "parsing_method")
                )
                or "regex"
            )

        if parsing_method == "regex":
            return self._find_template_blocks_by_regex(raw_str)
        elif parsing_method == "char":
            return self._find_template_blocks_by_char(raw_str)
        else:
            raise SQLFluffUserError(
                f"Invalid 'parsing_method' for dataform templater: {parsing_method}. "
                + "Expected 'regex' or 'char'."
            )

    def _find_template_blocks_by_char(self, raw_str: str) -> list[BlockSpan]:
        """
        Uses a brace-counting parser to find all `${...}`, `js {...}`,
        and `config {...}` blocks.
        """
        results: list[BlockSpan] = []

        templater_logger.debug("Finding template blocks in raw string.")
        starters: list[tuple[str, BlockType]] = [
            ("post_operations {", BlockType.POST_OPERATIONS),
            ("pre_operations {", BlockType.PRE_OPERATIONS),
            ("config {", BlockType.CONFIG),
            ("js {", BlockType.JS),
            ("${", BlockType.TEMPLATED),
        ]

        stack = 0
        outer_block_start = -1
        inner_content_start = -1
        block_type: Optional[BlockType] = None

        i = 0
        while i < len(raw_str):
            # if we're not in a block, look for a starter sequence
            if stack == 0:
                # try to match js blocks first (higher priority)
                matched = False
                for test_prefix, prefix_block_type in starters:
                    if raw_str.startswith(test_prefix, i):
                        stack = 1
                        block_type = prefix_block_type
                        outer_block_start = i
                        inner_content_start = i + len(test_prefix)
                        # jump index past starter, -1 for the loop increment
                        i += len(test_prefix) - 1
                        matched = True
                        break
                    if not matched:  # if no higher-priority block matched
                        # check for SQL block comments /* ... */
                        if raw_str.startswith("/*", i):
                            outer_start = i
                            comment_end_marker_idx = raw_str.find("*/", i + 2)

                            if comment_end_marker_idx != -1:
                                outer_end = comment_end_marker_idx + 2
                            else:
                                outer_end = len(raw_str)  # unclosed block comment

                            results.append(
                                BlockSpan(
                                    outer_start,
                                    outer_start,
                                    outer_end,
                                    outer_end,
                                    BlockType.SQL_BLOCK_COMMENT,
                                )
                            )
                            # adjust for the i += 1 at the end of loop
                            i = outer_end - 1

                        # check for SQL line comments -- ...
                        elif raw_str.startswith("--", i):
                            outer_start = i
                            newline_idx = raw_str.find("\n", i + 2)

                            if newline_idx != -1:
                                outer_end = newline_idx
                            else:
                                # Line comment goes to end of string
                                outer_end = len(raw_str)

                            results.append(
                                BlockSpan(
                                    outer_start,
                                    outer_start,
                                    outer_end,
                                    outer_end,
                                    BlockType.SQL_LINE_COMMENT,
                                )
                            )
                            # adjust for the i += 1 at the end of loop
                            i = outer_end - 1

            # if in a block, count all braces
            else:
                char = raw_str[i]
                if char == "{":
                    stack += 1
                elif char == "}":
                    stack -= 1
                    if stack == 0:
                        inner_content_end = i
                        outer_block_end = i + 1
                        assert block_type is not None
                        results.append(
                            BlockSpan(
                                outer_block_start,
                                inner_content_start,
                                inner_content_end,
                                outer_block_end,
                                block_type,
                            )
                        )
                        # reset for the next block
                        block_type = None
                        outer_block_start = -1
            i += 1
        templater_logger.debug(f"Found {len(results)} blocks.")
        return results

    def _annotate_sqlx_with_markers(
        self, in_str: str, fname: str, all_blocks: list[BlockSpan]
    ) -> str:
        """
        Annotates the SQLX file by wrapping templated blocks with IIFE markers.

        Returns the transformed SQLX string and a map from marker IDs to original
        inner content of the templated blocks.
        """
        transformed_parts: list[str] = []
        last_idx = 0
        templated_block_counter = 0

        templater_logger.info("Annotating sqlx file: %s", fname)
        for block_span in all_blocks:
            # add the literal part before the current block
            transformed_parts.append(in_str[last_idx : block_span.outer_start])

            if block_span.block_type == BlockType.TEMPLATED:
                original_inner_code = in_str[
                    block_span.inner_start : block_span.inner_end
                ]
                marker_id = MARKER_ID_FORMAT.format(templated_block_counter)

                wrapped_code = (
                    "${ (() => { "
                    + f"const __sqlfluff_tpl_out__ = ({original_inner_code}); "
                    + f"return `/*~START_MARKER:{marker_id}~*/${{__sqlfluff_tpl_out__}}/*~END_MARKER:{marker_id}~*/`; "  # noqa: E501
                    + "})() }"
                )
                transformed_parts.append(wrapped_code)
                templated_block_counter += 1
            else:
                # other blocks are unchanged
                transformed_parts.append(
                    in_str[block_span.outer_start : block_span.outer_end]
                )
            last_idx = block_span.outer_end

        # add any remaining literal part after the last block
        transformed_parts.append(in_str[last_idx:])
        transformed_sqlx = "".join(transformed_parts)

        return transformed_sqlx

    def _copy_project_files_to_temp_dir(self, project_dir: Path, temp_dir: Path):
        """Copies necessary Dataform project files to the temp compilation directory."""

        def copy_file(filename):
            copy_project_file = project_dir / filename
            if (copy_project_file).exists():
                shutil.copy2(copy_project_file, temp_dir / filename)
                templater_logger.debug(f"Copied {filename}")

        def copy_dir(dirname):
            copy_project_dir = project_dir / dirname
            if copy_project_dir.exists():
                shutil.copytree(copy_project_dir, temp_dir / dirname)
                templater_logger.debug(f"Copied {dirname}/ directory")

        copy_file("package.json")
        copy_file("package-lock.json")
        copy_file("dataform.json")
        copy_file("workflow_settings.yaml")
        copy_dir("includes")
        copy_dir("definitions")

    def _execute_dataform_compile(
        self, temp_dir: Path, config: Optional[FluffConfig]
    ) -> DataformCompilationResult:
        """
        Executes the `dataform compile` command and returns the parsed JSON output.
        """
        if (temp_dir / "package.json").exists():
            templater_logger.info("Found package.json, running `npm install`...")
            try:
                npm_executable = shutil.which("npm")
                if not npm_executable:
                    raise FileNotFoundError("npm not found")
                subprocess.run(
                    [npm_executable, "install", "--production"],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                npm_stderr: Optional[str] = None
                if isinstance(e, subprocess.CalledProcessError):
                    npm_stderr = e.stderr
                raise SQLFluffUserError(
                    "Failed to run `npm install`. "
                    + f"Is `npm` installed and on the PATH?\nError: {e}\n"
                    + f"Stderr: {npm_stderr if npm_stderr else 'N/A'}"
                ) from e

        try:
            dataform_executable = self._get_dataform_executable(config)
            found_dataform_executable = shutil.which(dataform_executable)
            if not found_dataform_executable:
                raise FileNotFoundError(
                    f'Dataform executable "{dataform_executable}" not found'
                )
            templater_logger.debug(
                f"Found dataform executable: {found_dataform_executable}"
            )
            templater_logger.info("Running `dataform compile`...")
            compile_process = subprocess.run(
                args=[Path(found_dataform_executable).absolute(), "compile", "--json"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            compile_output: DataformCompilationResult = json.loads(
                compile_process.stdout
            )
            templater_logger.debug(f"Dataform compile output: {compile_output}")
            return compile_output
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            json.JSONDecodeError,
        ) as e:
            stderr: Optional[str] = None
            if isinstance(e, subprocess.CalledProcessError):
                stderr = e.stderr

            raise SQLFluffUserError(
                "Failed to compile Dataform project. "
                + f"Is `{dataform_executable}` installed and on the PATH?\nError: {e}\n"
                + f"Stderr: {stderr if stderr else 'N/A'}"
            ) from e

    def _extract_compiled_query(
        self, compile_result: DataformCompilationResult, original_fname_path: Path
    ) -> str:
        """
        Extracts the compiled SQL query for the target file from the Dataform
        compilation result.
        """
        templater_logger.debug(
            f"Extracting compiled query for {original_fname_path.as_posix()}"
        )
        compiled_sql: Optional[str] = None
        compiled_objects = itertools.chain(
            ((obj, "table") for obj in compile_result.get("tables", [])),
            ((obj, "assertion") for obj in compile_result.get("assertions", [])),
            ((obj, "operation") for obj in compile_result.get("operations", [])),
        )
        for compiled_obj, obj_type in compiled_objects:
            output_file_name = Path(compiled_obj["fileName"])
            if output_file_name.as_posix() == original_fname_path.as_posix():
                if obj_type == "operation":
                    queries = cast(DataformCompiledOperation, compiled_obj).get(
                        "queries"
                    )
                    if queries:
                        compiled_sql = "".join(queries)
                else:
                    compiled_sql = cast(DataformCompiledTable, compiled_obj).get(
                        "query"
                    )

                break

        if not compiled_sql:
            raise SQLTemplaterError(
                f"Could not find compiled SQL for file {str(original_fname_path)!r} "
                + "in Dataform output."
            )
        templater_logger.debug(f"Extracted compiled SQL: {compiled_sql[:100]}...")
        return compiled_sql

    def _add_literal_slices(
        self,
        text: str,
        source_pos: int,
        templated_pos: int,
        raw_slices: list[RawFileSlice],
        templated_slices: list[TemplatedFileSlice],
    ) -> tuple[int, int]:
        """Helper to create and append slices for literal (non-templated) text."""
        if not text:
            return source_pos, templated_pos

        raw_slices.append(RawFileSlice(text, "literal", source_pos))
        templated_slices.append(
            TemplatedFileSlice(
                "literal",
                slice(source_pos, source_pos + len(text)),
                slice(templated_pos, templated_pos + len(text)),
            )
        )
        source_pos += len(text)
        templated_pos += len(text)
        return source_pos, templated_pos

    def _map_slices_from_compiled_sql(
        self, in_str: str, compiled_sql: str, all_blocks: list[BlockSpan]
    ) -> tuple[str, list[RawFileSlice], list[TemplatedFileSlice]]:
        """
        Maps the original source string and compiled SQL into raw and templated slices.
        """
        final_templated_str = MARKER_RE_PAIR.sub(r"\g<templated_content>", compiled_sql)

        raw_file_slices: list[RawFileSlice] = []
        templated_file_slices: list[TemplatedFileSlice] = []

        current_source_pos = 0  # tracks position in original 'in_str'
        current_templated_pos = 0  # tracks position in 'final_templated_str'

        compiled_marker_iter = MARKER_RE_PAIR.finditer(compiled_sql)
        next_compiled_marker_match = next(compiled_marker_iter, None)

        for block_span in all_blocks:
            # handle literal part before current block
            literal_text = in_str[current_source_pos : block_span.outer_start]
            current_source_pos, current_templated_pos = self._add_literal_slices(
                literal_text,
                current_source_pos,
                current_templated_pos,
                raw_file_slices,
                templated_file_slices,
            )

            # handle the current block
            original_block_full_text = in_str[
                block_span.outer_start : block_span.outer_end
            ]

            if block_span.block_type == BlockType.TEMPLATED:
                if not next_compiled_marker_match:
                    raise SQLTemplaterError(
                        "Mismatch in templated block count between original and compiled SQL. "  # noqa: E501
                        + "Expected more blocks in compiled output."
                    )

                templated_content_compiled = next_compiled_marker_match.group(
                    "templated_content"
                )
                original_outer_content = in_str[
                    block_span.outer_start : block_span.outer_end
                ]

                raw_file_slices.append(
                    RawFileSlice(
                        original_outer_content, "templated", block_span.outer_start
                    )
                )
                templated_file_slices.append(
                    TemplatedFileSlice(
                        "templated",
                        slice(block_span.outer_start, block_span.outer_end),
                        slice(
                            current_templated_pos,
                            current_templated_pos + len(templated_content_compiled),
                        ),
                    )
                )
                current_source_pos = block_span.outer_end
                current_templated_pos += len(templated_content_compiled)
                next_compiled_marker_match = next(compiled_marker_iter, None)
            elif block_span.block_type in (BlockType.CONFIG, BlockType.JS):
                raw_file_slices.append(
                    RawFileSlice(
                        original_block_full_text,
                        block_span.block_type.value,
                        block_span.outer_start,
                    )
                )
                # blocks are removed from templated output
                templated_file_slices.append(
                    TemplatedFileSlice(
                        block_span.block_type.value,
                        slice(block_span.outer_start, block_span.outer_end),
                        zero_slice(current_templated_pos),
                    )
                )
                current_source_pos = block_span.outer_end

        # handle trailing literal part (if any)
        remaining_literal_text = in_str[current_source_pos:]
        current_source_pos, current_templated_pos = self._add_literal_slices(
            remaining_literal_text,
            current_source_pos,
            current_templated_pos,
            raw_file_slices,
            templated_file_slices,
        )

        if current_source_pos != len(in_str):
            templater_logger.warning(
                "Source string length mismatch after slicing: "
                + f"tracked {current_source_pos} vs actual {len(in_str)}."
                + " This might lead to inaccurate linting positions."
            )
        if current_templated_pos != len(final_templated_str):
            templater_logger.warning(
                f"Templated string length mismatch after slicing: "
                f"tracked {current_templated_pos} vs actual {len(final_templated_str)}."
                + " This might lead to inaccurate linting positions."
            )
        if next_compiled_marker_match:
            templater_logger.warning(
                "Remaining templated markers in compiled SQL after processing all blocks. "  # noqa: E501
                + "This might indicate a mismatch in block identification."
            )

        return final_templated_str, raw_file_slices, templated_file_slices

    def _compile_files(
        self,
        fnames: list[str],
        project_dir: Path,
        config: Optional[FluffConfig] = None,
        override_fname: Optional[str] = None,
        override_content: Optional[str] = None,
    ):
        """
        Handles the Dataform compilation process for a given list of file paths.
        Caches the results internally.
        """
        if not project_dir.exists():
            raise SQLFluffUserError(
                "Could not find Dataform project root "
                + f"(dataform.json or workflow_settings.yaml) from path: {project_dir}"
            )

        templater_logger.info("Starting Dataform compilation...")
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            templater_logger.info("Created temporary project directory: %s", temp_dir)

            self._copy_project_files_to_temp_dir(project_dir, temp_dir)

            for fname in fnames:
                fname_path = Path(fname).resolve()

                if (
                    override_fname
                    and fname == override_fname
                    and override_content is not None
                ):
                    in_str = override_content
                else:
                    in_str = fname_path.read_text()

                relative_fname = fname_path.relative_to(project_dir)
                temp_fpath = temp_dir / relative_fname
                temp_fpath.parent.mkdir(parents=True, exist_ok=True)

                all_blocks = self._find_template_blocks(in_str, config)
                self._compilation_cache[fname] = CompilationCache(
                    blocks=all_blocks, compiled_sql=""
                )

                transformed_sqlx = self._annotate_sqlx_with_markers(
                    in_str, fname, all_blocks
                )
                _ = temp_fpath.write_text(transformed_sqlx)

            compile_result = self._execute_dataform_compile(temp_dir, config)

            for fname in fnames:
                fname_path = Path(fname).resolve()
                compiled_sql = self._extract_compiled_query(
                    compile_result, fname_path.relative_to(project_dir)
                )
                self._compilation_cache[fname].compiled_sql = compiled_sql
        templater_logger.info("Dataform compilation finished.")

    @override
    def sequence_files(
        self,
        fnames: list[str],
        config: Optional[FluffConfig] = None,
        formatter: Optional[FormatterInterface] = None,
    ) -> Iterable[str]:
        """Sequences the files for processing."""
        templater_logger.info("sequencing files: %s", fnames)

        self._sequenced_files = fnames
        self._compilation_cache = {}

        yield from fnames

    @override
    @large_file_check
    def process(
        self,
        *,
        in_str: str,
        fname: str,
        config: Optional[FluffConfig] = None,
        formatter: Optional[FormatterInterface] = None,
    ) -> tuple[TemplatedFile, list[SQLTemplaterError]]:
        """Annotate and process .sqlx file."""
        if not self._sequenced_files:
            self._sequenced_files = [fname]
            self._compilation_cache = {}

        # in-memory content differing from on-disk content?
        is_content_different = False
        fname_path = Path(fname)
        if fname_path.exists():
            on_disk_content = fname_path.read_bytes()
            in_str_content = in_str.encode("utf-8")

            on_disk_hash = hashlib.sha256(on_disk_content).hexdigest()
            in_str_hash = hashlib.sha256(in_str_content).hexdigest()

            if on_disk_hash != in_str_hash:
                is_content_different = True

        # 1. The file is not in the cache.
        # 2. The in-memory content is different from on-disk
        if fname not in self._compilation_cache or is_content_different:
            templater_logger.info(
                f"Recompiling... Cache empty: {fname not in self._compilation_cache}, "
                + f"Content changed: {is_content_different}"
            )
            project_dir = Path(self._get_project_dir(config))

            if is_content_different:
                templater_logger.info(
                    "File content mismatch, re-compiling with in-memory content."
                )

            override_fname = fname if is_content_different else None
            override_content = in_str if is_content_different else None
            self._compilation_cache = {}
            self._compile_files(
                self._sequenced_files,
                project_dir,
                config,
                override_fname=override_fname,
                override_content=override_content,
            )

        # retrieve compilation results from cache
        all_blocks = self._compilation_cache[fname].blocks
        compiled_sql = self._compilation_cache[fname].compiled_sql

        (
            final_templated_str,
            raw_file_slices,
            templated_file_slices,
        ) = self._map_slices_from_compiled_sql(in_str, compiled_sql, all_blocks)

        return (
            TemplatedFile(
                source_str=in_str,
                templated_str=final_templated_str,
                fname=fname,
                sliced_file=templated_file_slices,
                raw_sliced=raw_file_slices,
            ),
            [],
        )
