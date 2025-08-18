## Aggregate
---
Performs a Rolling Aggregate on the input data according to specifications.

Given:
- `df_artifact`: Input DataFrame
- `keys`: Labels to work by 
- `metrics`: List of metrics to perform the aggregate task on 
- `labels`: List of labels to perform the label aggregate on 
- `metric_aggs`: Array of aggregations to run on the metrics  
  **ex:** `['mean', 'sum']`
- `label_aggs`: Array of aggregations to run on the labels  
  **ex:** `['max']`
- `suffix`: A suffix to add to the aggregations (**ex:** '_daily')
- `window`: Window size, can be `int` for number of rows, or in time interval string if timestamp index is available 
- `center`: Perform aggregation on the sample at the window's center
- `inplace`: 
 - **True:** Returns only the newly aggregated data 
 - **False:** Returns the newly aggregated data with the original dataset