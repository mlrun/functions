# Modules

Modules are reusable building blocks you can import into your MLRun project. 
They can be generic utility code, but they can also be more specialized - such as a model-monitoring application or a wrapper around an application runtime.
As public contributions, we ask that all contributors follow the project’s guidelines and conventions (please chip in).

## Catalog

<!-- AUTOGEN:START (do not edit below) -->
| Name | Description | Kind | Categories |
| --- | --- | --- | --- |
| [agent_deployer](https://github.com/mlrun/functions/tree/development/modules/src/agent_deployer) | Helper for serving function deploy of an AI agents using MLRun | monitoring_application | model-serving |
| [count_events](https://github.com/mlrun/functions/tree/development/modules/src/count_events) | Count events in each time window | monitoring_application | model-serving |
| [evidently_iris](https://github.com/mlrun/functions/tree/development/modules/src/evidently_iris) | Demonstrates Evidently integration in MLRun for data quality and drift monitoring using the Iris dataset | monitoring_application | model-serving, structured-ML |
| [histogram_data_drift](https://github.com/mlrun/functions/tree/development/modules/src/histogram_data_drift) | Model-monitoring application for detecting and visualizing data drift | monitoring_application | model-serving, structured-ML |
| [openai_proxy_app](https://github.com/mlrun/functions/tree/development/modules/src/openai_proxy_app) | OpenAI application runtime based on fastapi | generic | genai |
| [vllm_module](https://github.com/mlrun/functions/tree/development/modules/src/vllm_module) | Deploys a vLLM OpenAI-compatible LLM server as an MLRun application runtime, with configurable GPU usage, node selection, tensor parallelism, and runtime flags. | generic | genai |
<!-- AUTOGEN:END -->
