from importlib import import_module
from inspect import getfullargspec, FullArgSpec
from sklearn.utils.testing import all_estimators

import sklearn
skversion = sklearn.__version__

def _create_class(pkg_class: str):
    """Create a class from a package.module.class string

    :param pkg_class:  full class location,
                       e.g. "sklearn.model_selection.GroupKFold"
    """
    splits = pkg_class.split(".")
    clfclass = splits[-1]
    pkg_module = splits[:-1]
    class_ = getattr(import_module(".".join(pkg_module)), clfclass)
    return class_

def _create_function(pkg_func: list):
    """Create a function from a package.module.function string

    :param pkg_func:  full function location,
                      e.g. "sklearn.feature_selection.f_classif"
    """
    splits = pkg_func.split(".")
    pkg_module = ".".join(splits[:-1])
    cb_fname = splits[-1]
    pkg_module = __import__(pkg_module, fromlist=[cb_fname])
    function_ = getattr(pkg_module, cb_fname)
    return function_

def get_model_configs(
    my_models: Union[str, List[str]],
    class_key = "CLASS",
    fit_key = "FIT",
    meta_key = "META",
) -> Union[dict, List[dict]]:
    """build sklearn model configuration parameters
    
    Take (full) class name of an scikit-learn model 
    and retrieve its `class` and `fit` parameters and
    their default values.
    
    Also returns some useful metadata values for the class
    """
    # get a list of all sklearn estimators
    estimators = all_estimators()
    def _get_estimator(pkg_class):
        """find a specific class in a list of sklearn estimators"""
        my_class = pkg_class.split(".")[-1]
        return list(filter(lambda x: x[0] == my_class, estimators))[0]

    # find estimators corresponding to my_models list
    my_estimators = []
    my_models = [my_models] if isinstance(my_models, str) else my_models
    for model in my_models:
        estimator_name, estimator_class = _get_estimator(model)
        my_estimators.append((estimator_name, estimator_class))

    # get class and fit specs
    estimator_specs = []
    for an_estimator in my_estimators:
        estimator_specs.append((an_estimator[0], # model only name
                                getfullargspec(an_estimator[1]), # class params
                                getfullargspec(an_estimator[1].fit), # fit params
                                an_estimator[1])) # package.module.model

    model_configs = []

    for estimator in estimator_specs:
        model_json = {class_key: {}, fit_key: {}}
        fit_params = {}

        for i, key in enumerate(model_json.keys()):
            f = estimator[i+1]
            args_paired = []
            defs_paired = []

            # reverse the args since there are fewer defaults than args
            args = f.args
            args.reverse()
            n_args = len(args)

            defs = f.defaults
            if defs is None:
                defs = [defs]
            defs = list(defs)
            defs.reverse()
            n_defs = len(defs)

            n_smallest = min(n_args, n_defs)
            n_largest = max(n_args, n_defs)

            # build 2 lists that can be concatenated
            for ix in range(n_smallest):
                if args[ix] is not "self":
                    args_paired.append(args[ix])
                    defs_paired.append(defs[ix])

            for ix in range(n_smallest, n_largest):
                if ix is not 0 and args[ix] is not "self":
                    args_paired.append(args[ix])
                    defs_paired.append(None)
               # concatenate lists into appropriate structure
            model_json[key] = dict(zip(reversed(args_paired), reversed(defs_paired)))

        model_json[meta_key] = {}
        model_json[meta_key]["sklearn_version"] = skversion
        model_json[meta_key]["class"] = ".".join([estimator[3].__module__, estimator[0]])
        model_configs.append(model_json)
    if len(model_configs) == 1:
        # do we want to log this modified model as an artifact?
        return model_configs[0]
    else:
        # do we want to log this modified model as an artifact?
        return model_configs

def update_model_config(
    config: dict,
    new_class: dict,
    new_fit: dict,
    class_key: str = "CLASS",
    fit_key: str = "FIT"
):
    """Update model config json
    
    Not used until we refactor as per the TODO
        
    This function is essential since there are modifications in class
    and fit params that must be made (callbacks are a good example, without
    which there is no training history available)
    
    TODO:  currently a model config contains 2 keys, but this will likely
    expand to include other functions beyond class and fit. So need to expand 
    this to a list of Tuple(str, dict), where `str` corresponds to a key
    in the model config and `dict` contains the params and their new values.
    
    :param config:      original model definition containing 2 keys, CLASS and FIT
    :param new_class:   new class key-values
    :param new_fit:     new fit key-values
    """
    config[class_key].update(new_class)
    config[fit_key].update(new_fit)
    
    return config