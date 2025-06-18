import numpy as np
import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt


def get_shap_values(
    model: xgb.Booster,
    Xtrain: pd.DataFrame,
    Xvalidation: pd.DataFrame,
    N_SHAP_BACKGROUND_SAMPLES: int = 200,
    N_SHAP_EXPLAINED_SAMPLES: int = 100,
) -> np.ndarray:
    background_data = shap.kmeans(X=Xtrain, k=N_SHAP_BACKGROUND_SAMPLES).data
    explainer = shap.TreeExplainer(
        model=model,
        data=background_data,
        feature_names=Xtrain.columns.tolist(),
        feature_perturbation="auto",
    )

    validation_indices = np.random.choice(
        len(Xvalidation), size=N_SHAP_EXPLAINED_SAMPLES, replace=False
    )
    validation_data = Xvalidation.iloc[validation_indices]

    shap_values = explainer.shap_values(validation_data)
    return shap_values


def plot_shap_summary(
    shap_values: np.ndarray, Xtrain: pd.DataFrame, Xvalidation: pd.DataFrame
) -> None:
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        Xvalidation,
        feature_names=Xtrain.columns.tolist(),
        show=False,
    )
    plt.tight_layout()
    plt.savefig("images/shap_summary.png", bbox_inches="tight", dpi=300)
    plt.close()
