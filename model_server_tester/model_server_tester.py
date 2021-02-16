import os
import pandas as pd
import requests
import json
import numpy as np

from datetime import datetime
from mlrun.datastore import DataItem
from mlrun.artifacts import get_model, ChartArtifact


def model_server_tester(
    context,
    table: DataItem,
    addr: str,
    label_column: str = "label",
    model: str = "",
    match_err: bool = False,
    rows: int = 20,
):
    """Test a model server

    :param table:         csv/parquet table with test data
    :param addr:          function address/url
    :param label_column:  name of the label column in table
    :param model:         tested model name
    :param match_err:     raise error on validation (require proper test set)
    :param rows:          number of rows to use from test set
    """

    table = table.as_df()

    y_list = table.pop(label_column).values.tolist()
    context.logger.info(f"testing with dataset against {addr}, model: {model}")
    if rows and rows < table.shape[0]:
        table = table.sample(rows)

    count = err_count = match = 0
    times = []
    for x, y in zip(table.values, y_list):
        count += 1
        event_data = json.dumps({"instances": [x.tolist()]})
        had_err = False
        try:
            start = datetime.now()
            resp = requests.put(f"{addr}/{model}/predict", json=event_data)
            if not resp.ok:
                context.logger.error(f"bad function resp!!\n{resp.text}")
                err_count += 1
                continue
            times.append((datetime.now() - start).microseconds)

        except OSError as err:
            context.logger.error(f"error in request, data:{event_data}, error: {err}")
            err_count += 1
            continue

        y_resp = resp.json()[0]
        if y == y_resp:
            match += 1

    context.log_result("total_tests", count)
    context.log_result("errors", err_count)
    context.log_result("match", match)
    if count - err_count > 0:
        times_arr = np.array(times)
        context.log_result("avg_latency", int(np.mean(times_arr)))
        context.log_result("min_latency", int(np.amin(times_arr)))
        context.log_result("max_latency", int(np.amax(times_arr)))

        chart = ChartArtifact("latency", header=["Test", "Latency (microsec)"])
        for i in range(len(times)):
            chart.add_row([i + 1, int(times[i])])
        context.log_artifact(chart)

    context.logger.info(
        f"run {count} tests, {err_count} errors and {match} match expected value"
    )

    if err_count:
        raise ValueError(f"failed on {err_count} tests of {count}")

    if match_err and match != count:
        raise ValueError(f"only {match} results match out of {count}")
