from mlrun import code_to_function
import os

def test_gen_class_data():
    fn = code_to_function(name='test_gen_class_data',
                          filename="gen_class_data.py",
                          handler="gen_class_data",
                          kind="job"
                          )
    fn.run(params={
            "n_samples": 10_000,
            "m_features": 5,
            "k_classes": 2,
            "weight": [0.5, 0.5],
            "sk_params": {"n_informative": 2},
            "file_ext": "csv"}, local=True, artifact_path="./artifacts/inputs")

    assert(os.path.exists("./artifacts/inputs/classifier-data.csv"))