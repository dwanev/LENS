import copy
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse

from _serializable import Deserializer, Serializable, Serializer


class Explainer(Serializable):
    """Uses Shapley values to explain any machine learning model or python function.

    This is the primary explainer interface for the SHAP library. It takes any combination
    of a model and masker and returns a callable subclass object that implements
    the particular estimation algorithm that was chosen.
    """

    def __init__(self, model, **kwargs):
        """Build a new explainer for the passed model.

        Parameters
        ----------
        model : object or function
            User supplied function or model object that takes a dataset of samples and
            computes the output of the model for those samples.
        pass

       """
        self.model = model

    def __call__(self, X, y=None):
        """Explains the output of model(*args), where args is a list of parallel iterable datasets.

        Note this default version could be an abstract method that is implemented by each algorithm-specific
        subclass of Explainer. Descriptions of each subclasses' __call__ arguments
        are available in their respective doc-strings.
        """
        start_time = time.time()

        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = getattr(self, "data_feature_names", None)

        # generate a np array, contribution of each feature, for each example
        v = self.shap_values(X, y=y, from_call=True, check_additivity=check_additivity, approximate=self.approximate)

