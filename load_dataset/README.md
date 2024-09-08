# load to datasets

Loads a scikit-learn toy dataset for classification or regression
    
The following datasets are available ('name' : desription):

    'iris'            : iris dataset (classification)
    'wine'            : wine dataset (classification)
    'breast_cancer'   : breast cancer wisconsin dataset (classification)
    'digits'          : digits dataset (classification)
    'boston'          : boston house-prices dataset (regression)
    'diabetes'        : diabetes dataset (regression)
    'linnerud'        : linnerud dataset (multivariate regression)

Currently the `iris`, `wine` and `breast_cancer` datasets run through the sklearn_classifier.  

**TODO**: The digits dataset requires the addition of images (add one or two lines of code to flatten the image pixel matrix to a feature vector, maybe add a parameter to indicate the inputs are images and the final image size for input to ML algo, maybe create a separate image preprocessing stage that can run processing in parallel and feed the trainer from a queue, trainer blocks until queue starts to fill...)

**TODO**: The regression datasets are available through this function, however a `sklearn_regression` function needs to be written, almost a copy paste of `sklearn_classifier`.  Alternatively, the training function can be split into 2 parts, fit and evaluate, where the fit is identical for regressor or classifier, and only the evaluate differs. 

The scikit-learn toy dataset functions return a data bunch including the following items:<br>
&emsp;{<br>
&emsp;&emsp;'data'  :  the features matrix,<br>
&emsp;&emsp;'target' : the ground truth labels<br>
&emsp;&emsp;'DESCR'  :  a description of the dataset<br>
&emsp;&emsp;'feature_names' :  header for data<br>
&emsp;}<br>

The features (and their names) are stored with the target labels in a DataFrame.

For further details see **[Scikit Learn Toy Datasets](https://scikit-learn.org/stable/datasets/index.html#toy-datasets)**