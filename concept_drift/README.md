# Concept Drift

**Concept drift** is a change in the statistical properties of the **target variable** over time.

When deploying our models to production, we must ensure our models perform as we expect them to - reaching the same level of performence we have seen on our test sets or at least performing in the same quality as when they were deployed.

However, often this is not the case. there are many factors that can affect our model's performance like seasonality or any unkown root causes that will change the laws underlying our data and invalidate some assumptions made by the model.

We offer this function to help combat Concept Drift with implementation of streaming DDM, EDDM and PH concept drift detectors.

## How to integrate

This function is made of two parts:

1. Kubernetes job to instantiate the selected models with a provided base dataset (the test dataset could be used)
2. [Nuclio serverless function](../concept_drift_streaming/concept_drift_streaming.ipynb) listed on a _labeled stream_, which will be deployed from this function after the models initialization and run the models per event and provide necessary alerts.

There are two steps to integrate sucessfully with your workflow:

1. Provide a stream where each event containes the joined **label** and **prediction** for that specific event.
2. Add this function to the workflow with the following params:

```markdown
:param context:         MLRun context
:param base_dataset:    Dataset containing label_col and prediction_col to initialize the detectors
:param input_stream:    labeled stream to track.
                        Should contain label_col and prediction_col
:param output_stream:   Output stream to push the detector's alerts
:param output_tsdb:     Output TSDB table to allow analysis and display
:param tsdb_batch_size: Batch size of alerts to buffer before pushing to the TSDB
:param callbacks:       Additional rest endpoints to send the alert data to
:param models:          List of the detectors to deploy
                        Defaults to ['ddm', 'eddm', 'pagehinkley'].
:param models_dest:     Location for saving the detectors
                        Defaults to 'models' (in relation to artifact_path).
:param pagehinkley_threshold:  Drift level threshold for PH detector Defaults to 10.
:param ddm_warning_level:      Warning level alert for DDM detector Defaults to 2.
:param ddm_out_control_level:  Drift level alert for DDM detector Defaults to 3.
:param label_col:       Label column to be used on base_dataset and input_stream
                        Defaults to 'label'.
:param prediction_col:  Prediction column to be used on base_dataset and input_stream
                        Defaults to 'prediction'.
:param hub_url:         hub_url in case the default is not used, concept_drift_streaming will be loaded
                        by this url
                        Defaults to mlconf.hub_url.
:param fn_tag:          hub tag to use
                        Defaults to 'master'
```

## Algorithms

We offer to deploy up to 3 concept drift streaming detectors

### DDM - Drift Detection Method

Models the **Number of errors** as a **binomial** variable. This enables us to confine the expected number of errors in a prediction stream window to within some standard deviation.

- Good for **abrupt** drift changes

<center>

![$mu=np_t$](https://latex.codecogs.com/svg.latex?mu=np_t)

![$\sigma=\sqrt{\frac{p_t(1-p_t)}{n}}$](<https://latex.codecogs.com/svg.latex?\sigma=\sqrt{\frac{p_t(1-p_t)}{n}}>)

</center>

**Alert** when:

<center>

![$p_t+\sigma_t\ge{p_{min}+3\sigma_{min}}$](https://latex.codecogs.com/svg.latex?p_t+\sigma_t\ge{p_{min}+3\sigma_{min}})

</center>

### EDDM - Early Drift Detection Method

Uses the distance between two consecutive errors.

- works better for **gradual** drift changes.
- More sensitive then DDM for noise
- Requires Minimal number of errors to initialize the statistics.

**Warning**:

<center>

![$\frac{p_t+2\sigma_t}{p_{max}+2\sigma_{max}}<0.95$](https://latex.codecogs.com/svg.latex?\frac{p_t+2\sigma_t}{p_{max}+2\sigma_{max}}<0.95)

</center>

**Alert**:

<center>

![$\frac{p_t+2\sigma_t}{p_{max}+2\sigma_{max}}<0.90$](https://latex.codecogs.com/svg.latex?\frac{p_t+2\sigma_t}{p_{max}+2\sigma_{max}}<0.90)

</center>

### PageHinkley Test:

The PageHinkley test is a sequential analysis technique typically used for monitoring change detection. (The test was designed to detect change in avg. of a Gaussian signal). In this test we use:  
x*1*, ..., x*n* - labeled dataset  
δ - magnitude threshold  
λ - detection threshold

<center>

![$\hat{x_T}=\frac{1}{T}\sum_{t=1}^{t}{x_t}$](https://latex.codecogs.com/svg.latex?\hat{x_T}=\frac{1}{T}\sum_{t=1}^{t}{x_t})

![$\sum_{t=1}^T{x_t-\hat{x_T}-\delta}$](https://latex.codecogs.com/svg.latex?U_T=\sum_{t=1}^T{x_t-\hat{x_T}-\delta})

![$m_T=min(U_t,t=1..T)$](<https://latex.codecogs.com/svg.latex?m_T=min(U_t,t=1..T)>)

</center>

**Alert**:

<center>

![$U_T-m_T>\lambda$](https://latex.codecogs.com/svg.latex?U_T-m_T>\lambda)

</center>

## Additional resources
[A Study on Change Detection Methods](https://pdfs.semanticscholar.org/bb6e/8a44c0efcd725aae1c0b1817561f6e278c2c.pdf), Raquel Sebasti˜ao1,2 and Jo˜ao Gama1,3, 1 LIAAD-INESC Porto L.A., University of Porto
Rua de Ceuta, 118 - 6, 4050-190 Porto, Portugal
2 Faculty of Science, University of Porto
3 Faculty of Economics, University of Porto
{raquel,jgama}@liaad.up.pt

[MLOps Live #4 - How to Detect & Remediate Drift in Production with MLOps Automation](https://www.youtube.com/watch?v=66_Q7mJZOSc&t=1296s)
