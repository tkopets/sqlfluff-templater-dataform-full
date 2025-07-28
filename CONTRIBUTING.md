# Contributing

Contributions are welcome! Please follow these guidelines to contribute to the project.

## Development

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/sqlfluff-templater-dataform-full.git
    ```
2.  Create a virtual environment and install the dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .[dev]
    ```
3.  Run the formatter:
    ```bash
    poe format
    ```
4.  Run the checks:
    ```bash
    poe check
    ```
5.  Submit a pull request with your changes.

### Troubleshooting

If you encounter a `ModuleNotFoundError` after installing dependencies, try deactivating and reactivating your virtual environment to refresh the shell's command cache:

```bash
deactivate
source .venv/bin/activate
```

## Releasing a New Version

To release a new version of this package, follow these steps:

1.  **Update the Version**: In `pyproject.toml`, increment the `version` number following [Semantic Versioning](https://semver.org/) guidelines.
2.  **Commit the Version Change**:
    ```bash
    git add pyproject.toml
    git commit -m "Bump version to X.Y.Z"
    ```
3.  **Tag the Release**: Create a new Git tag that matches the version in `pyproject.toml`.
    ```bash
    git tag vX.Y.Z
    ```
4.  **Push to GitHub**: Push the commit and the new tag to the repository.
    ```bash
    git push
    git push origin vX.Y.Z
    ```
5.  **Publish the Release**: GitHub Actions workflow will automatically build and publish the package to PyPI.
