# Contributing To MLRun's Hub

## Types of Assets You Can Contribute
- Modules (either a generic module or a model monitoring application)
- Functions (will be converted to MLRun Runtime)

## How to Contribute
1. Fork this repository on GitHub and create a new branch for your new asset.
2. Add a new directory for your asset under the appropriate directory (`functions/src` for functions, `modules/src` for modules).
3. Populate the directory with your asset files (see the [Asset Structure](#asset-structure) section below).
4. Open a pull request to merge your changes into the main repository, to its **development** branch.

## Asset Structure
### Functions
```txt
functions
├── src
│   ├── your_function_name
│   │   ├── item.yaml
│   │   ├── function.yaml
│   │   ├── your_function_name.py
│   │   ├── your_function_name.ipynb
│   │   ├── test_your_function_name.py 
│   │   └── requirements.txt
```
- `item.yaml`: Metadata about the function. Can be generated using the following CLI command:
  ```bash
    python -m cli.cli generate-item-yaml function your_function_name
  ```
  Then, fill in all the relevant details. Important: Be consistent with the module name across the directory name, all relevant `item.yaml` fields, and the file names.

- `function.yaml`: The MLRun function definition. Can be generated from `item.yaml` using:
  ```bash
    python -m cli.cli item-to-function --item-path functions/src/your_function_name
    ```
- `your_function_name.py`: The main code file for your function. (Notice: keep the code well-documented, the docstrings are used in the hub UI as documentation for the function.)
- `your_function_name.ipynb`: A Jupyter notebook demonstrating the function's usage. (Notice: the notebook must be able to run end-to-end automatically without manual intervention.)
- `test_your_function_name.py`: Unit tests for your function. (Will run upon each change to your function).
- `requirements.txt`: Any additional Python dependencies required by your function's unit tests. (Notice: The function's own dependencies should be specified in the `item.yaml` file, not here.)

### Modules
```txt
modules
├── src
│   ├── your_module_name
│   │   ├── item.yaml
│   │   ├── your_module_name.py
│   │   ├── your_module_name.ipynb
│   │   ├── test_your_module_name.py 
│   │   └── requirements.txt
```
- `item.yaml`: Metadata about the module. Can be generated using the following CLI command:
  ```bash
    python -m cli.cli generate-item-yaml module your_module_name
  ```
  Then, fill in all the relevant details. Important: Be consistent with the module name across the directory name, all relevant `item.yaml` fields, and the file names.
- `your_module_name.py`: The main code file for your module. (Notice: keep the code well-documented, the docstrings are used in the hub UI as documentation for the module.)
- `your_module_name.ipynb`: A Jupyter notebook demonstrating the module's usage.
- `test_your_module_name.py`: Unit tests for your module. (Will run upon each change to your module).
- `requirements.txt`: Any additional Python dependencies required by your module's unit tests. (Notice: The module's own dependencies should be specified in the `item.yaml` file, not