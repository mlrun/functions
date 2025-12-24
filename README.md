# MLRun Functions Hub

A centralized repository for open-source MLRun functions, modules, and steps that can be used as reusable components in ML pipelines.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Make Commands](#make-commands)
- [CLI Commands](#cli-commands)
- [Contributing](#contributing)
- [Testing](#testing)
- [Code Standards](#code-standards)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10 or 3.11** - Required
- **UV** - Fast Python package manager (required)
- **Git** - For version control
- **Make** (optional) - For convenient command shortcuts

> **Note:** This project uses UV as the baseline package manager. All dependency management and installation is done through UV.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mlrun/functions.git
   cd functions
   ```

2. **Install dependencies:**
   ```bash
   uv sync --all-groups --prerelease=allow
   ```

> **Note:** UV will automatically create a virtual environment if one doesn't exist. Make sure to activate it with `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows).

## Make Commands

The project includes a Makefile for convenient command shortcuts:

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make sync` | Sync dependencies from lockfile |
| `make format` | Format code with black and isort |
| `make lint` | Run code linters |
| `make test FUNC=<name>` | Run tests for a specific function |
| `make cli ARGS="<args>"` | Run CLI with custom arguments |

### Examples

```bash
# Sync dependencies
make sync

# Format code
make format

# Run tests for aggregate function
make test FUNC=aggregate

# Run CLI command
make cli ARGS="generate-item-yaml function my_function"
```

## CLI Commands

The project includes a CLI tool for managing MLRun functions, modules, and steps.

### Usage

```bash
python -m cli.cli [COMMAND] [OPTIONS]
```

### Available Commands

#### 1. generate-item-yaml
Generate an `item.yaml` file from a Jinja2 template.

**Syntax:**
```bash
python -m cli.cli generate-item-yaml TYPE NAME
```

**Examples:**
```bash
# Generate item.yaml for a function
python -m cli.cli generate-item-yaml function aggregate

# Generate item.yaml for a module
python -m cli.cli generate-item-yaml module my_module

# Generate item.yaml for a step
python -m cli.cli generate-item-yaml step my_step
```

#### 2. item-to-function
Create a `function.yaml` file from an `item.yaml` file.

**Syntax:**
```bash
python -m cli.cli item-to-function --item-path PATH
```

**Example:**
```bash
python -m cli.cli item-to-function --item-path functions/src/aggregate
```

#### 3. function-to-item
Create an `item.yaml` file from a `function.yaml` file.

**Syntax:**
```bash
python -m cli.cli function-to-item --path PATH
```

**Example:**
```bash
python -m cli.cli function-to-item --path functions/src/aggregate
```

#### 4. run-tests
Run the test suite for a specific asset.

**Syntax:**
```bash
python -m cli.cli run-tests -r PATH -s TYPE -fn NAME
```

**Example:**
```bash
python -m cli.cli run-tests -r functions/src/aggregate -s py -fn aggregate
```

#### 5. build-marketplace
Build and create a PR for the marketplace directory.

**Syntax:**
```bash
python -m cli.cli build-marketplace [OPTIONS]
```

#### 6. update-readme
Update README files with auto-generated content.

**Syntax:**
```bash
python -m cli.cli update-readme [OPTIONS]
```

## Contributing

We welcome contributions! Follow these steps to contribute:

### Types of Assets You Can Contribute

- **Functions** - MLRun runtime functions (job or serving)
- **Modules** - Generic modules or model monitoring applications
- **Steps** -  MLRun steps to be used in serving graphs

### How to Contribute

1. **Fork the repository** on GitHub
2. **Create a new branch** for your asset:
   ```bash
   git checkout -b feature/my-new-function
   ```
3. **Add your asset** under the appropriate directory:
   - Functions: `functions/src/`
   - Modules: `modules/src/`
   - Steps: `steps/src/`
4. **Follow the asset structure** (see below)
5. **Test your asset** thoroughly
6. **Format your code**:
   ```bash
   make format
   ```
7. **Open a pull request** to the **development** branch

### Asset Structure

#### Functions

```txt
functions/src/your_function_name/
├── item.yaml                      # Metadata (required)
├── function.yaml                  # MLRun function definition (required)
├── your_function_name.py          # Main code file (required)
├── your_function_name.ipynb       # Demo notebook (required)
├── test_your_function_name.py     # Unit tests (required)
└── requirements.txt               # Test dependencies (optional)
```

**Steps to create a function:**

1. Generate the item.yaml template:
   ```bash
   python -m cli.cli generate-item-yaml function your_function_name
   ```

2. Fill in the `item.yaml` with:
   - `kind`: `job`, `serving`, or `nuclio:serving`
   - `categories`: Browse [MLRun hub](https://www.mlrun.org/hub/functions/) for existing categories
   - Other metadata fields

3. Generate the function.yaml:
   ```bash
   python -m cli.cli item-to-function --item-path functions/src/your_function_name
   ```

4. Implement your function in `your_function_name.py`:
   - Keep code well-documented (docstrings are used in the hub UI)
   - Specify function dependencies in `item.yaml`, not in requirements.txt

5. Create a demo notebook (`your_function_name.ipynb`):
   - Must run end-to-end automatically
   - Demonstrate the function's usage

6. Write unit tests (`test_your_function_name.py`):
   - Cover functionality as much as possible
   - Tests run automatically on each change

#### Modules

```txt
modules/src/your_module_name/
├── item.yaml                      # Metadata (required)
├── your_module_name.py            # Main code file (required)
├── your_module_name.ipynb         # Demo notebook (required)
├── test_your_module_name.py       # Unit tests (required)
└── requirements.txt               # Test dependencies (optional)
```

**Steps to create a module:**

1. Generate the item.yaml:
   ```bash
   python -m cli.cli generate-item-yaml module your_module_name
   ```

2. Fill in the `item.yaml` with:
   - `kind`: `generic` or `monitoring_application`
   - `categories` and other metadata

3. Implement, document, and test your module

For model monitoring modules, see the [MLRun model monitoring guidelines](https://docs.mlrun.org/en/stable/model-monitoring/applications.html).

### Contribution Checklist

- [ ] Asset follows the proper directory structure
- [ ] `item.yaml` is complete and accurate
- [ ] Code is well-documented with docstrings
- [ ] Demo notebook runs end-to-end without errors
- [ ] Unit tests cover the functionality
- [ ] Code is formatted with `black` and `isort`
- [ ] All tests pass locally
- [ ] PR targets the **development** branch

## Testing

### Running Tests

**Test a specific function:**
```bash
make test FUNC=aggregate
# or
python -m cli.cli run-tests -r functions/src/aggregate -s py -fn aggregate
```

**Run tests manually with pytest:**
```bash
cd functions/src/aggregate
pytest test_aggregate.py -v
```

### Writing Unit Tests

- Place tests in `test_<asset_name>.py`
- Use pytest as the testing framework
- Mock external dependencies when necessary
- Test edge cases and error conditions
- Ensure tests are reproducible
- Note: Tests will be run automatically on each change in the CI pipeline

**Example test structure:**
```python
import pytest
from your_function_name import your_function

def test_basic_functionality():
    result = your_function(param1="value1")
    assert result is not None
    assert result.status == "success"

def test_error_handling():
    with pytest.raises(ValueError):
        your_function(invalid_param="bad_value")
```

## Code Standards

### Python Style Guide

We follow **PEP 8** style guidelines with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Sorted with isort
- **Docstrings**: Google style or NumPy style
- **Type hints**: Encouraged for function signatures

### Formatting Tools

**Black** - Code formatter:
```bash
make format
# or
uv run black .
```

**isort** - Import sorter:
```bash
uv run isort .
```

**Run both:**
```bash
make format
```

### Linting

Check code quality without modifying files:
```bash
make lint
# or
uv run black --check .
uv run isort --check-only .
```

### Documentation Standards

- **Docstrings are mandatory** for all public functions, classes, and modules
- Use clear, concise descriptions
- Include parameter types and return types
- Provide usage examples when helpful

**Example:**
```python
def train_model(data: pd.DataFrame, target_column: str, model_type: str = "sklearn") -> dict:
    """
    Train a machine learning model on the provided dataset.
    
    Args:
        data: Input DataFrame containing features and target
        target_column: Name of the target column
        model_type: Type of model to train (default: "sklearn")
    
    Returns:
        Dictionary containing the trained model and metrics
        
    Example:
        >>> result = train_model(df, "label", "sklearn")
        >>> print(result["accuracy"])
        0.95
    """
    # Implementation here
```

## Troubleshooting

### Common Issues

#### CLI Commands Not Working

**Problem:** `python -m cli.cli` fails

**Solution:**
```bash
# Check if you're in the right directory
pwd  # Should be the project root

# Ensure dependencies are installed
make install

# Try running with full path
python -m cli.cli --help
```

#### Tests Failing

**Problem:** Tests fail when running locally

**Solution:**
```bash
# Install test dependencies if the function has a requirements.txt
cd functions/src/your_function
uv pip install -r requirements.txt

# Run tests with verbose output
pytest test_your_function.py -v -s

# Check for missing environment variables or configuration
```


### Getting Help

If you encounter issues:

1. Check the [MLRun documentation](https://docs.mlrun.org/)
2. Search [GitHub issues](https://github.com/mlrun/functions/issues)
3. Open a new issue with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version, UV version)

## Resources

- **MLRun Hub UI**: https://www.mlrun.org/hub/
- **MLRun Documentation**: https://docs.mlrun.org/
- **MLRun Marketplace Repository**: http://github.com/mlrun/marketplace
- **MLRun Community**: https://github.com/mlrun/mlrun

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support:
- Open an issue on [GitHub](https://github.com/mlrun/functions/issues)
- Join the MLRun community discussions
- Check the [MLRun documentation](https://docs.mlrun.org/)

---

**Happy Contributing! 🚀**

