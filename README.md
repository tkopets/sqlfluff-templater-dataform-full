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

1.  **Identify Blocks**: The plugin first identifies different types of blocks in your `.sqlx` file: Dataform templated SQL (`${...}`), JavaScript blocks (`js {...}`), and configuration blocks (`config {...}`).
2.  **Insert Markers**: For Dataform templated SQL blocks (`${...}`), the content is temporarily wrapped with unique, invisible markers within an Immediately Invoked Function Expression (IIFE). JavaScript and config blocks are passed through largely unchanged.
3.  **Compile with Dataform CLI**: A temporary Dataform project is created, relevant files are copied, and the `dataform compile` command is executed on the transformed `.sqlx` file. **The Dataform CLI must be installed and accessible in your system's PATH for this plugin to function.**
4.  **Map Slices**: After compilation, the plugin parses the compiled output. It uses the inserted markers to accurately map the compiled SQL back to its original positions in the `.sqlx` source file. This allows SQLFluff to report linting and formatting issues at the correct locations.

## Configuration

You can configure the templater by adding the following options to your `.sqlfluff` file under the `[sqlfluff:templater:dataform-full]` section.

- **`project_dir`**: Specifies the path to your Dataform project root. If not provided, the templater will search for a Dataform project in the current working directory.

- **`dataform_executable`**: Sets a custom path to the Dataform executable. This is useful if the executable is not in your system's `PATH`. This setting takes precedence over the `DATAFORM_EXECUTABLE` environment variable.

- **`parsing_method`**: This templater offers two different methods for parsing `.sqlx` files before compilation. "Parsing" here means identifying the different blocks like `config {...}`, `js {...}`, and `${...}`.
  - **`regex` (Default)**: A fast parser that uses regular expressions. Recommended for most use cases.
  - **`char`**: A character-by-character parser that is more resilient to complex or unusual nesting of blocks.

### Example Configuration

```ini
[sqlfluff:templater:dataform-full]
# Path to your Dataform project
project_dir = path/to/your/dataform/project

# Custom path to the Dataform executable
dataform_executable = /path/to/your/dataform_cli

# To use the character-based parser instead of the default regex parser
# parsing_method = char
```

## Development

This plugin follows the standard SQLFluff plugin development guide.
The core logic resides in `src/sqlfluff_templater_dataform/templater.py`, specifically the `process` method.

## Known Issues

- The templater relies on balanced curly braces (`{}`) for identifying `js {}` and `config {}` blocks. Unbalanced braces within these blocks (e.g., unmatched `{` or `}`) will cause parsing errors during templating and lead to incorrect linting. Ensure all such blocks have a matching number of opening and closing braces.
