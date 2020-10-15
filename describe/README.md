# describe

Get the table's summary statistics and summary plots

The functions will require the following parameters:

'''

:param context:         the function context
:param table:           MLRun input pointing to pandas dataframe (csv/parquet file path)
:param label_column:    ground truth column label
:param class_labels:    label for each class in tables and plots
:param plot_hist:       (True) set this to False for large tables
:param plots_dest:      destination folder of summary plots (relative to artifact_path)
:param update_dataset:  when the table is a registered dataset update the charts in-place

'''

The function will output the following artifacts per column within the data frame (based on data types):

1. histogram chart
2. violin chart
3. imbalance chart
4. correlation-matrix chart
5. correlation-matrix csv
6. imbalance-weights-vec csv