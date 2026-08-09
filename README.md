# Dataform Templater Plugin for SQLFluff

This plugin integrates [SQLFluff](https://www.sqlfluff.com/) with Dataform projects, allowing SQLFluff to lint and format your Dataform SQLX files after templating.

## Getting Started

1.  **Install the plugin**:

    ```bash
    pip install sqlfluff-templater-dataform-full
    ```

2.  **Configure SQLFluff**:
    Add the following to your `.sqlfluff` configuration file:

    ```ini
    [sqlfluff]
    dialect = bigquery
    templater = dataform-full
    sql_file_exts = .sql,.sqlx
    ```

3.  **Usage**:
    Run SQLFluff as usual:
    ```bash
    sqlfluff lint your_dataform_project/
    ```

## How it works

This templater operates by using the Dataform CLI to compile `.sqlx` files. It performs the following steps:

1.  **Identify Blocks**: The plugin first identifies the different types of block in your `.sqlx` file: Dataform templated SQL (`${...}`), JavaScript (`js {...}`), configuration (`config {...}`), `pre_operations {...}`, `post_operations {...}`, `input "..." {...}`, and SQL comments.
2.  **Insert Markers**: For Dataform templated SQL blocks (`${...}`), the content is temporarily wrapped with unique, invisible markers within an Immediately Invoked Function Expression (IIFE). JavaScript and config blocks are passed through largely unchanged.
3.  **Compile with Dataform CLI**: A temporary Dataform project is created, relevant files are copied, and the `dataform compile` command is executed on the transformed `.sqlx` file. **The Dataform CLI must be installed for this plugin to function** — either on your `PATH`, or pointed at via the `dataform_executable` setting below.
4.  **Map Slices**: After compilation, the plugin parses the compiled output. It uses the inserted markers to accurately map the compiled SQL back to its original positions in the `.sqlx` source file. This allows SQLFluff to report linting and formatting issues at the correct locations.

## Configuration

You can configure the templater by adding the following options to your `.sqlfluff` file under the `[sqlfluff:templater:dataform-full]` section.

- **`project_dir`**: Specifies the path to your Dataform project root. If not provided, the templater will search for a Dataform project in the current working directory.

- **`dataform_executable`**: Sets a custom path to the Dataform executable. This is useful if the executable is not in your system's `PATH`. This setting takes precedence over the `DATAFORM_EXECUTABLE` environment variable.

- **`parsing_method`**: This templater offers two methods for parsing `.sqlx` files before compilation. "Parsing" here means identifying the different blocks like `config {...}`, `js {...}`, and `${...}`.
  - **`lexer` (Default)**: A context-tracking lexer, compatible with [Dataform's own SQLX lexer](https://github.com/dataform-co/dataform/blob/main/sqlx/lexer.ts). It tracks SQL and JS strings, template literals and comments, so a brace, `${` or comment marker *inside* a string or comment cannot mis-terminate a block (`js { const x = "}"; }` and a `${...}` inside a SQL comment are both handled correctly). Recommended for all use cases.

    It follows Dataform's rules even where they surprise: block openers need exactly one space before the brace (`config {`, not `config  {`), `input` names are limited to `[a-zA-Z0-9_-]`, and what Dataform does not model — JS regex literals, a bare `{` inside a `pre_operations`/`post_operations`/`input` body — is mis-sliced the same way here. On a file it cannot finish, it warns and falls back to `regex`.
  - **`regex`**: A best-effort parser built on regular expressions, kept as a fallback. It is not string- or comment-aware, so it can mis-slice blocks containing braces or `${` inside strings or comments; it also misses `config` blocks nested more than three levels deep and `input` blocks with several names, because Python regexes cannot recurse. A mis-sliced file usually fails with a length mismatch and is skipped from linting, so prefer the default.

### Example Configuration

```ini
[sqlfluff:templater:dataform-full]
# Path to your Dataform project
project_dir = path/to/your/dataform/project

# Custom path to the Dataform executable
dataform_executable = /path/to/your/dataform_cli

# Parser used to identify blocks (default: lexer). Override only if needed:
# parsing_method = regex
```

## Development

This plugin follows the standard SQLFluff plugin development guide.
The core logic resides in `sqlfluff_templater_dataform_full/templater.py`, specifically the `process` method.

The end-to-end tests shell out to the Dataform CLI, so it must be on your `PATH`:

```bash
PATH="path/to/dataform_project/node_modules/.bin:$PATH" pytest test/
```

## Known Issues

- Block detection follows Dataform's own lexer, which does not model JavaScript regex literals. A `/` regex containing a `}` — as in `js { const re = /}/; }` — ends the block early, both here and in Dataform itself.
- A genuinely unbalanced brace inside a `config {}` or `js {}` block still breaks templating, and will lead to incorrect linting. Braces inside strings, comments and template literals are fine — those only tripped up the older `regex` parser.
