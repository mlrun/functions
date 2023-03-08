# Great Expectations Validation
![Great Expectations Logo](doc/great-expectations-logo-full-size.png)

Run data validation via Great Expectations. Will validate a given dataset with a given set of expectations, run the validation, and log the output HTML data doc in MLRun.

## Prerequisites

See [1_set_expectations.ipynb](1_set_expectations.ipynb) for a full example.

- Initialized a Great Expectations project
- Configured at least one Datasource i.e. `my_datasource`
- Created at least one Expectation Suite i.e. `my_suite`
- Created a Checkpoint i.e. `my_checkpoint`

## Usage

See [2_validate_expectations.ipynb](2_validate_expectations.ipynb) for a full example.

```python
import mlrun

fn = mlrun.import_function("hub://great_expectations")
run = fn.run(
    inputs={"data": "https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv"},
    params={
        "expectation_suite_name": "test_suite",
        "data_asset_name": "iris_dataset",
    },
)
```

## All Configuration
Inputs
```rst
:param data:                   Data to validate. Can be local or remote link.
```

Parameters
```rst
:param expectation_suite_name: Name of expectation suite to validate against.
:param data_asset_name:        Name of dataset in Great Expectations.
:param datasource_name:        Name of datasource to use for validation.
:param data_connector_name:    Name of data connector to use for validation.
:param datasource_config:      Full configuration for datasource. For use with custom
                               data sources other than the default pandas datasource.
:param batch_identifiers:      Custom metadata for identifying particular batches of
                               data. For use when not using the default batch identifiers.
:param root_directory:         Path to underlying Great Expectations project. Defaults to
                               MLRun project artifact path if not specified.
:param checkpoint_name:        Name of checkpoint to use for validation.
:param checkpoint_config:      Full configuration for checkpoint. For use with custome
                               checkpoint config other than the default.
```