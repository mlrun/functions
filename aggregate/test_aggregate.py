from mlrun import code_to_function, new_task

METRICS_PATH = 'data/metrics.pq'


def test_run_local_aggregate():

    fn = code_to_function(name='test_aggregate',
                          filename="aggregate.py",
                          handler="aggregate",
                          kind="job",
                         )
    fn.run(params={'metrics': ['cpu_utilization'],
                                     'labels': ['is_error'],
                                     'metric_aggs': ['mean', 'sum'],
                                     'label_aggs': ['max'],
                                     'suffix': 'daily',
                                     'inplace': False,
                                     'window': 5,
                                     'center': True,
                                     'save_to': 'aggregate.pq',
                                     'files_to_select': 2}
            , local=True
            , inputs={'df_artifact': METRICS_PATH}
            )
