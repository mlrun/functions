# MLRun Hub

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
- [Resources](#resources)

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10 or 3.11 (recommended) ** - Required
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
   git checkout -b feature/my-new-item
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
   - `kind`: `job` or `serving`
   - `categories`: Browse [MLRun hub](https://www.mlrun.org/hub/functions/) for existing categories
   - `version`, `description`, and other metadata

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
   - `version`, `description`, etc.

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

- **Docstrings are mandatory** for all public hub items
- Use clear, concise descriptions
- Include parameter types and return types
- Provide usage examples when helpful

**Example (function `auto_trainer`):**
```python
def train(
    context: MLClientCtx,
    dataset: DataItem,
    model_class: str,
    label_columns: Optional[Union[str, List[str]]] = None,
    drop_columns: List[str] = None,
    model_name: str = "model",
    tag: str = "",
    sample_set: DataItem = None,
    test_set: DataItem = None,
    train_test_split_size: float = None,
    random_state: int = None,
    labels: dict = None,
    **kwargs,
):
    """
    Training a model with the given dataset.

    example::

        import mlrun
        project = mlrun.get_or_create_project("my-project")
        project.set_function("hub://auto_trainer", "train")
        trainer_run = project.run(
            name="train",
            handler="train",
            inputs={"dataset": "./path/to/dataset.csv"},
            params={
                "model_class": "sklearn.linear_model.LogisticRegression",
                "label_columns": "label",
                "drop_columns": "id",
                "model_name": "my-model",
                "tag": "v1.0.0",
                "sample_set": "./path/to/sample_set.csv",
                "test_set": "./path/to/test_set.csv",
                "CLASS_solver": "liblinear",
            },
        )

    :param context:                 MLRun context
    :param dataset:                 The dataset to train the model on. Can be either a URI or a FeatureVector
    :param model_class:             The class of the model, e.g. `sklearn.linear_model.LogisticRegression`
    :param label_columns:           The target label(s) of the column(s) in the dataset. for Regression or
                                    Classification tasks. Mandatory when dataset is not a FeatureVector.
    :param drop_columns:            str or a list of strings that represent the columns to drop
    :param model_name:              The model's name to use for storing the model artifact, default to 'model'
    :param tag:                     The model's tag to log with
    :param sample_set:              A sample set of inputs for the model for logging its stats along the model in favour
                                    of model monitoring. Can be either a URI or a FeatureVector
    :param test_set:                The test set to train the model with.
    :param train_test_split_size:   if test_set was provided then this argument is ignored.
                                    Should be between 0.0 and 1.0 and represent the proportion of the dataset to include
                                    in the test split. The size of the Training set is set to the complement of this
                                    value. Default = 0.2
    :param random_state:            Relevant only when using train_test_split_size.
                                    A random state seed to shuffle the data. For more information, see:
                                    https://scikit-learn.org/stable/glossary.html#term-random_state
                                    Notice that here we only pass integer values.
    :param labels:                  Labels to log with the model
    :param kwargs:                  Here you can pass keyword arguments with prefixes,
                                    that will be parsed and passed to the relevant function, by the following prefixes:
                                    - `CLASS_` - for the model class arguments
                                    - `FIT_` - for the `fit` function arguments
                                    - `TRAIN_` - for the `train` function (in xgb or lgbm train function - future)

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
make sync

```

#### Tests Failing

**Problem:** Tests fail when running locally

**Solution:**
```bash
# Install test dependencies if the item has a requirements.txt
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

