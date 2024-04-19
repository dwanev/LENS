import xgboost
import shap
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from shap._explanation import Explanation

import lens_xai

import src.necsuf_tabular_text as necsuf_tabular_text
from src.recourse_experiment import deg_nec_suff

def example_waterfall_plot():
    """
    Display a waterfall plot on synthetic data.
    :return:
    """
    exp = Explanation(
        np.array([ [0.7,-0.5] ]),
        base_values=np.array([ [0.0] ]),
        data=np.array([ [5,-5] ]),
        feature_names=["good","bad"],
    )
    shap.plots.waterfall(exp[0], max_display=10, show=True)



def shap_example(model, X):
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0])


def lens_example(model, X):



    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = lens_xai.Explainer(model)
    shap_values = explainer(X)

    # visualize the first prediction's explanation
    lens_xai.plots.waterfall(shap_values[0])



def lens_example_2():
    #
    # Abbreviations
    # CF - CounterFactual
    # SCM - Structural Causal Model
    # i2r - input-to-reference - for examining the sufficiency/necessity of a contrastive model prediction
    # r2i - reference-to-input - for examining the sufficiency/necessity of the original model prediction
    #

    german_cred_df = pd.read_csv("./datasets/german_credit_data.csv")
    # following standard pre-processing from https://www.kaggle.com/vigneshj6/german-credit-data-analysis-python
    german_cred_df['Saving accounts'] = german_cred_df['Saving accounts'].map(
        {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3});
    german_cred_df['Saving accounts'] = german_cred_df['Saving accounts'].fillna(
        german_cred_df['Saving accounts'].dropna().mean())
    german_cred_df['Checking account'] = german_cred_df['Checking account'].map(
        {"little": 0, "moderate": 1, "rich": 2});
    german_cred_df['Checking account'] = german_cred_df['Checking account'].fillna(
        german_cred_df['Checking account'].dropna().mean())
    german_cred_df['Sex'] = german_cred_df['Sex'].map({"male": 0, "female": 1});
    german_cred_df['Housing'] = german_cred_df['Housing'].map({"own": 0, "free": 1, "rent": 2});
    german_cred_df['Purpose'] = german_cred_df['Purpose'].map(
        {'radio/TV': 0, 'education': 1, 'furniture/equipment': 2, 'car': 3, 'business': 4,
         'domestic appliances': 5, 'repairs': 6, 'vacation/others': 7});
    german_cred_df['Risk'] = german_cred_df['Risk'].map({"good": 1, "bad": 0});
    german_cred_df.rename(columns={"Risk": "outcome", "Saving accounts": "Savings",
                                   "Checking account": "Checking", "Credit amount": "Credit"}, inplace=True)
    german_cred_df.drop("Unnamed: 0", axis=1, inplace=True)

    inp = pd.DataFrame(german_cred_df.iloc[-1]).T
    num_features = len(inp.columns[:-1])

    # SCM model
    SCM_models = necsuf_tabular_text.fit_scm(german_cred_df)

    # Tree
    X, y = np.array(german_cred_df.iloc[:, :-1]), np.array(german_cred_df.iloc[:, -1:]).ravel()
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, test_size=.2, random_state=42)
    clf = ExtraTreesClassifier(random_state=0, max_depth=15)
    clf.fit(X_train, y_train)
    f_inp = clf.predict(np.array(inp.iloc[:, :-1]))

    # Notice this time we use the causal_SCM argument, and pass in the SCM we fitted above
    _, CF_i2r_causal, refs1_causal = \
        necsuf_tabular_text.suff_nec_pipeline((german_cred_df.outcome != inp.outcome.item()), inp, clf, german_cred_df,
                                  num_features, causal_SCM=SCM_models, n_sample=100,
                                  col_con=[0, 6, 7], col_cat=[1, 2, 3, 4, 5, 8])

    pass

    sub_df = deg_nec_suff(CF_i2r_causal, inp, f_inp, clf, num_features,
                                     r2i=False, deg_thresh=0, datatype='Tabular',
                                     filter_supersets=True, filter_cost=True,
                                     pred_on_fly=True, max_output=5)
    pass
    sub_df.columns
    sub_df.degree # tuples of index and value
    sub_df.cost # tuples of index and value

    exp = Explanation(
        np.array([ sub_df.degree.values ]),  # contribution of each feature
        base_values=np.array([ [0.0] ]), # average or 'base' results
        # data=np.array([ [5,-5] ]), # info values?
        feature_names=sub_df.columns.values, # feature names
    )
    shap.plots.waterfall(exp[0], max_display=10, show=True)
    pass


if __name__ == "__main__":
    # train an XGBoost model
    X, y = shap.datasets.california()
    model = xgboost.XGBRegressor().fit(X, y)

    # shap_example(model, X)

    lens_example_2()

    # example_waterfall_plot()
