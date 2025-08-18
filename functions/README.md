# Functions hub 

This functions hub is intended to be a centralized location for open source contributions of function components.  
These are functions expected to be run as independent mlrun pipeline compnents, and as public contributions, 
it is expected that contributors follow certain guidelines/protocols (please chip-in).

## Catalog

<!-- AUTOGEN:START (do not edit below) -->
| Name | Description | Kind | Categories |
| --- | --- | --- | --- |
| [aggregate](/home/runner/work/functions/functions/functions/src/aggregate) | Rolling aggregation over Metrics and Lables according to specifications | job | data-preparation |
| [arc_to_parquet](/home/runner/work/functions/functions/functions/src/arc_to_parquet) | retrieve remote archive, open and save as parquet | job | utils |
| [auto_trainer](/home/runner/work/functions/functions/functions/src/auto_trainer) | Automatic train, evaluate and predict functions for the ML frameworks - Scikit-Learn, XGBoost and LightGBM. | job | machine-learning, model-training |
| [azureml_serving](/home/runner/work/functions/functions/functions/src/azureml_serving) | AzureML serving function | serving | machine-learning, model-serving |
| [azureml_utils](/home/runner/work/functions/functions/functions/src/azureml_utils) | Azure AutoML integration in MLRun, including utils functions for training models on Azure AutoML platfrom. | job | model-serving, utils |
| [batch_inference](/home/runner/work/functions/functions/functions/src/batch_inference) | Batch inference (also knows as prediction) for the common ML frameworks (SciKit-Learn, XGBoost and LightGBM) while performing data drift analysis. | job | model-serving |
| [batch_inference_v2](/home/runner/work/functions/functions/functions/src/batch_inference_v2) | Batch inference (also knows as prediction) for the common ML frameworks (SciKit-Learn, XGBoost and LightGBM) while performing data drift analysis. | job | model-serving |
| [describe](/home/runner/work/functions/functions/functions/src/describe) | describe and visualizes dataset stats | job | data-analysis |
| [describe_dask](/home/runner/work/functions/functions/functions/src/describe_dask) | describe and visualizes dataset stats | job | data-analysis |
| [describe_spark](/home/runner/work/functions/functions/functions/src/describe_spark) |  | job | data-analysis |
| [feature_selection](/home/runner/work/functions/functions/functions/src/feature_selection) | Select features through multiple Statistical and Model filters | job | data-preparation, machine-learning |
| [gen_class_data](/home/runner/work/functions/functions/functions/src/gen_class_data) | Create a binary classification sample dataset and save. | job | data-generation |
| [github_utils](/home/runner/work/functions/functions/functions/src/github_utils) | add comments to github pull request | job | utils |
| [hugging_face_serving](/home/runner/work/functions/functions/functions/src/hugging_face_serving) | Generic Hugging Face model server. | serving | genai, model-serving |
| [load_dataset](/home/runner/work/functions/functions/functions/src/load_dataset) | load a toy dataset from scikit-learn | job | data-preparation |
| [mlflow_utils](/home/runner/work/functions/functions/functions/src/mlflow_utils) | Mlflow model server, and additional utils. | serving | model-serving, utils |
| [model_server](/home/runner/work/functions/functions/functions/src/model_server) | generic sklearn model server | nuclio:serving | model-serving, machine-learning |
| [model_server_tester](/home/runner/work/functions/functions/functions/src/model_server_tester) | test model servers | job | monitoring, model-serving |
| [noise_reduction](/home/runner/work/functions/functions/functions/src/noise_reduction) | Reduce noise from audio files | job | data-preparation, audio |
| [onnx_utils](/home/runner/work/functions/functions/functions/src/onnx_utils) | ONNX intigration in MLRun, some utils functions for the ONNX framework, optimizing and converting models from different framework to ONNX using MLRun. | job | utils, deep-learning |
| [open_archive](/home/runner/work/functions/functions/functions/src/open_archive) | Open a file/object archive into a target directory | job | utils |
| [pii_recognizer](/home/runner/work/functions/functions/functions/src/pii_recognizer) | This function is used to recognize PII in a directory of text files | job | data-preparation, NLP |
| [pyannote_audio](/home/runner/work/functions/functions/functions/src/pyannote_audio) | pyannote's speech diarization of audio files | job | deep-learning, audio |
| [question_answering](/home/runner/work/functions/functions/functions/src/question_answering) | GenAI approach of question answering on a given data | job | genai |
| [send_email](/home/runner/work/functions/functions/functions/src/send_email) | Send Email messages through SMTP server | job | utils |
| [silero_vad](/home/runner/work/functions/functions/functions/src/silero_vad) | Silero VAD (Voice Activity Detection) functions. | job | deep-learning, audio |
| [sklearn_classifier](/home/runner/work/functions/functions/functions/src/sklearn_classifier) | train any classifier using scikit-learn's API | job | machine-learning, model-training |
| [sklearn_classifier_dask](/home/runner/work/functions/functions/functions/src/sklearn_classifier_dask) | train any classifier using scikit-learn's API over Dask | job | machine-learning, model-training |
| [structured_data_generator](/home/runner/work/functions/functions/functions/src/structured_data_generator) | GenAI approach of generating structured data according to a given schema | job | data-generation, genai |
| [test_classifier](/home/runner/work/functions/functions/functions/src/test_classifier) | test a classifier using held-out or new data | job | machine-learning, model-testing |
| [text_to_audio_generator](/home/runner/work/functions/functions/functions/src/text_to_audio_generator) | Generate audio file from text using different speakers | job | data-generation, audio |
| [tf2_serving](/home/runner/work/functions/functions/functions/src/tf2_serving) | tf2 image classification server | nuclio:serving | model-serving, machine-learning |
| [transcribe](/home/runner/work/functions/functions/functions/src/transcribe) | Transcribe audio files into text files | job | audio, genai |
| [translate](/home/runner/work/functions/functions/functions/src/translate) | Translate text files from one language to another | job | genai, NLP |
| [v2_model_server](/home/runner/work/functions/functions/functions/src/v2_model_server) | generic sklearn model server | serving | model-serving, machine-learning |
| [v2_model_tester](/home/runner/work/functions/functions/functions/src/v2_model_tester) | test v2 model servers | job | model-testing, machine-learning |
<!-- AUTOGEN:END -->
