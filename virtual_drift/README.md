# Drift Magnitude

Concept drift and shift are major issues that greatly affect the accuracy and reliability of many real-world applications of machine learning.  We can use the following Drift Magnitude metrics to map and understand our concepts and how close the properties of the data we used to train the models on are to the current data we receive.

## How to integrate

The Virtual Drift function is built to receive two data batches of data (as `dataitem` or `Dataframe`), base batch $u$ and current batch $t$.  

```markdown
:param context:   MLRun context
:param t:         Base dataset for the drift metrics
:param u:         Test dataset for the drift metrics
:param label_col: Label colum in t and u
:param prediction_col: Predictions column in t and u
:param discritizers:   Dictionary of dicsritizers for the features if available
                       (Created automatically if not provided)
:param n_bins:    Number of bins to be used for histrogram creation from continuous variables
:param stream_name: Output stream to push metrics to
:param results_tsdb_container: TSDB table container to push metrics to
:param results_tsdb_table: TSDB table to push metrics to
```

The function will calculate the selected drift mangitude metrics that were selected and apply them to the **features**, **labels** and **predictions**.  It will then save those metrics and export them via Parquet and TSDB.

## Metrics

The drift magnitude metrics we calculate are:

### TVD - Total Variation Distance

Provides a symetric drift distance between two periods $u$ and $t$  
$Z$ - vector of random variables  
$P_t$ - Probability distribution over timespan $t$  

$\sigma_{t, u}(Z)=\frac{1}{2}\sum_{\hat{z}\in{dom(Z)}}{|P_t{(\hat{Z})-P_u{(\hat{Z})}}|}$

### Helinger Distance

Hellinger distance is an $f$ divergence measuer, similar to the Kullback-Leibler (KL) divergence. However, unlike KL Divergence the Hellinger divergence is symmetric and bounded over a probability space.

$P, Q$ - Discrete probability distributions ($p_i, ..., p_k$)

$H(P,Q)=\frac{1}{\sqrt{2}}\sqrt{\sum_{i=1}^{k}{(\sqrt{p_i}-\sqrt{q_i})^2}}$

### KL Divergence

KL Divergence (or relative entropy) is a measure of how one probability distribution differs from another.  It is an asymmetric measure (thus it's not a metric) and it doesn't satisfy the triangle inequality. KL Divergence of 0, indicates two identical distributrions.

$D_{KL}(P||Q)=\sum_{x\in{X}}{(P(x)\log{\frac{P(x)}{Q(x)}})}$

## Additional Resources

Webb, Geoffrey I. et al. “[Characterizing Concept Drift.](https://arxiv.org/abs/1511.03816)” Data Mining and Knowledge Discovery 30.4 (2016): 964–994. Crossref. Web.

[MLOps Live #4 - How to Detect & Remediate Drift in Production with MLOps Automation](https://www.youtube.com/watch?v=66_Q7mJZOSc&t=1296s)