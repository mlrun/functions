from mlrun import code_to_function

DATA_URL = "https://s3.wasabisys.com/iguazio/data/market-palce/arc_to_parquet/higgs-sample.csv.gz"

def test_run_local_arc_to_parquet():
    fn = code_to_function(name='test_arc_to_parquet',
                          filename="arc_to_parquet.py",
                          handler="arc_to_parquet",
                          kind="local",
                          )
    fn.spec.command = "arc_to_parquet.py"
    fn.run(params={"key": "higgs-sample"},
           handler="arc_to_parquet",
           inputs={"archive_url": DATA_URL},
           artifact_path='artifacts'
           #, local=True

           )



