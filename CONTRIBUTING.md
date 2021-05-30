# Contributing

+1tada First off, thanks for taking the time to read the guidelines and for considering contributing! tada+1

The following is a set of guidelines for contributing to the functions' repository and its packages. These are mostly 
guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

#### Table Of Contents
1) [What Should I Know Before I Start?](#what-should-i-know-before-i-start)
    1) [Concepts](#concepts)
        1) [Function](#1-functions)
        1) [function.yaml](#2-functionyaml)
        1) [Marketplace](#3-marketplace)
        1) [item.yaml](#4-itemyaml)
    1) [Function directory structure](#function-directory-structure)
    1) [item.yaml anatomy](#itemyaml-anatomy)
1) [Installation Guide](#installation-guide)
1) [Creating A New Function](#creating-a-new-function)
1) [Updating An Existing Function](#updating-an-existing-function)
1) [Testing Functions](#testing-functions)
    1) [item.yaml validation](#1-itemyaml-validation)
    1) [function.yaml validation](#2-functionyaml-validation)
    1) [Python unittests](#3-python-unittests)
    1) [Testing example notebooks](#4-testing-example-notebooks)
1) [The functions CLI](#command-line-utility)

## What Should I Know Before I Start?
### Concepts

#### 1) Functions
All the executions in MLRun are based on Serverless Functions, the functions allow specifying code and 
all the operational aspects (image, required packages, cpu/mem/gpu resources, storage, environment, etc.), the 
different function runtimes take care of automatically transforming the code and spec to fully managed and elastic 
services over Kubernetes which save significant operational overhead, address scalability and reduce infrastructure costs.

MLRun supports batch functions (based on Kubernetes jobs, Spark, Dask, Horovod, etc.) or Real-time functions for 
serving, APIs, and stream processing (based on the high-performance Nuclio engine).

Further reading: <br>
[MLRun docs](https://docs.mlrun.org/en/latest/runtimes/functions.html) <br>
[Function runtimes](https://docs.mlrun.org/en/latest/runtimes/functions.html#function-runtimes) <br>
[Nuclio docs](https://nuclio.io/docs/latest/) <br>

#### 2) function.yaml
A structure of configuration resembles the Kubernetes resource definitions, and includes the 
apiVersion, kind, metadata, spec, and status sections. This file is the result of running:
```python
import mlrun
fn = mlrun.code_to_function(...)
fn.export()
``` 

Further reading: <br>
[Code to function](https://docs.mlrun.org/en/latest/api/mlrun.html?highlight=code_to_function#mlrun.code_to_function)
[Function configuration](https://nuclio.io/docs/latest/reference/function-configuration/function-configuration-reference/) <br>
[Deploying functions](https://nuclio.io/docs/latest/tasks/deploying-functions/) <br>

#### 3) Marketplace
The function marketplace is a user-friendly representation of the `mlrun/functions` repository. The 
main purpose of the Function Marketplace is to provide a simple (yet interactive) and explorable interface for users to
find, filter and discover MLRun functions. It is partially inspired by the helm’s ArtifactHub.

Further reading: <br>
[Visit Function Marketplace](https://mlrun.github.io/marketplace/) <br>
[Helm Chart Artifact Hub](https://artifacthub.io/) <br>

#### 4) item.yaml
A structure of configuration that enables:
1) Generating a `function.yaml` without introducing any configuration in the example notebook
1) Allows the function to be listed on the marketplace 


### Function directory structure
This is a suggested function structure, deviating from this structure template will cause issues with marketplace 
rendering.
```text
<FUNCTION_NAME>
      |
      |__ <FUNCTION_NAME>.py (Containing the implementation code of FUNCTION_NAME)
      |
      |__ <FUNCTION_NAME>.ipynb (Containing an example for running and deploying the FUNCTION_NAME)
      |
      |__ item.yaml (Containing the spec for generating function.yaml and being listed on the marketplace)
      |
      |__ function.yaml (Containing the spec for deploying the function)
      |
      |__ test_<FUNCTION_NAME>.py (optional)
      |
      |__ test_<FUNCTION_NAME>.ipynb (optional)
```

### item.yaml anatomy
```yaml
apiVersion: v1
categories: []         # List of category names
description: ''        # Short description
example: ''            # Path to examole notebook
generationDate:        # Automatically created when creating a new item using the cli
icon: ''               # Path to icon file
labels: {}             # Key values label pairs
maintainers: []        # List of maintainers
mlrunVersion: ''       # Function’s MLRun version requirement, should follow python’s versioning schema
name: ''               # Function name
platformVersion: ''    # Function’s Iguazio version requirement, should follow python’s versioning schema
spec:
  filename: ''         # Implementation file
  handler: ''          # Handler function name
  image: ''            # Base image name
  kind: ''             # Function kind
  requirements: []     # List of Pythonic library requirements
  customFields: {}     # Key value pairs of custom spec fields
  env: []              # Spec environment params
version: 0.0.1         # Function version, should follow the standard semantic versioning schema
```

## Installation Guide
It is highly advised using the functions package in a dedicated environment, since Pipenv is used as part of the testing routine, 
conda can be used instead.

1) Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (or any other environment manager)
1) Clone the repository `git clone https://github.com/mlrun/functions.git`
1) cd to the functions directory `cd functions`
1) Create a new environment `conda install -n functions python=3.8`
1) Activate the environment `source activate functions`
1) Install the requirements `pip install -r requirements.txt`

## Creating A New Function
See [command line utility > section 6](#command-line-utility)  
See [function directory structure](#function-directory-structure)  
See [testing functions](#testing-functions)

## Updating An Existing Function
1) Fork the `mlrun/functions` [repository](https://github.com/mlrun/functions)
1) Open a branch with a name describing the function that is being changed, and what was changed  

    \* Make sure to update the version of the function in the `item.yaml`<br>
    \* If any business logic changed, make to update the `function.yaml` by running the 
`python function.py item-to-function [OPTIONS]` command <br>
1) Submit a PR


## Testing Functions
(WORK IN PROGRESS)
### 1) item.yaml validation
### 2) function.yaml validation
### 3) Python unittests
### 4) Testing example notebooks

## Command Line Utility
The command line utility supports multiple sub-commands:

1. Help

```text
python functions.py --help
```

```text
Usage: functions.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  build-marketplace
  create-legacy-catalog
  function-to-item
  item-to-function
  new-item
  run-tests

```
2. Build Functions build-marketplace

```text
python functions.py build-docs
```
```text
Usage: functions.py build-marketplace [OPTIONS]

Options:
  -s, --source-dir TEXT       Path to the source directory
  -sn, --source-name TEXT     Name of source, if not provided, name of source directory will be used instead (optional)
  -m, --marketplace-dir TEXT  Path to marketplace directory
  -T, --temp-dir TEXT         Path to intermediate build directory (optional)
  -c, --channel TEXT          Name of build channel
  -v, --verbose               When this flag is set, the process will output extra information (optional)
  --help                      Show this message and exit.
```

3. Create Legacy Catalog

```text
python functions.py create-legacy-catalog
```

```text
Usage: functions.py create-legacy-catalog [OPTIONS]

Options:
  -r, --root-dir TEXT  Path to root project directory
  --help               Show this message and exit.
```
4. Item To Function

```text
python functions.py item-to-function
```

```text
Usage: functions.py item-to-function [OPTIONS]

Options:
  -i, --item-path TEXT    Path to item.yaml file or a directory containing one
  -o, --output-path TEXT  Path to code_to_function output, will use item-path directory if not provided (optional)
  -c, --code_output       If spec.filename or spec.example is a notebook, should a python file be created (optional)
  -fmt, --format_code     If -c/--code_output is enabled, and -fmt/--format is enabled, the code output will be 
                          formatted by black formatter (optional)
  --help                  Show this message and exit.

```

5. Function To Item
```text
python functions.py function-to-item
```
```text
Usage: functions.py function-to-item [OPTIONS]

Options:
  -p, -path TEXT  Path to one of: specific function.yaml, directory containing function.yaml or a root directory to 
                  search function.yamls in
  --help          Show this message and exit.
```

6. New Item
```text
python functions.py new-item
```
```text
Usage: functions.py new-item [OPTIONS]

Options:
  -p, --path TEXT  Path to directory in which a new item.yaml will be created
  -o, --override   Override if already exists
  --help           Show this message and exit.
```
This sub command will create a directory (if doesn't exist already) with a copy the item.yaml template.
`-o/--override` can be used to override an existing item.yaml.

6. Test Suite
```text
python functions.py run-tests
```
```text
  -r, --root-directory TEXT     Path to root directory
  -s, --suite TEXT              Type of suite to run [py/ipynb/examples/items]
  -mp, --multi-processing TEXT  run multiple tests
  -fn, --function-name TEXT     run specific function by name
  -f, --stop-on-failure         When set, the test entire test run will fail once a single test fails
  --help                        Show this message and exit.
```