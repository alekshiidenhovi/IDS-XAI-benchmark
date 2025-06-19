import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def get_shap_values(
    explainer: shap.TreeExplainer,
    Xvalidation: pd.DataFrame,
    yvalidation: pd.Series,
    N_SHAP_EXPLAINED_SAMPLES: int = 100,
) -> np.ndarray:
    validation_indices = np.random.choice(
        len(Xvalidation), size=N_SHAP_EXPLAINED_SAMPLES, replace=False
    )
    x_val_samples = Xvalidation.iloc[validation_indices]
    y_val_samples = yvalidation.iloc[validation_indices]

    shap_values = explainer.shap_values(x_val_samples, y_val_samples)
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
