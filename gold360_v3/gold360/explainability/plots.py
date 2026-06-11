from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class ExplanationPlots:
    @staticmethod
    def global_importance_bar(importance_df: pd.DataFrame, top_n: int = 15,
                               title: str = "Global Feature Importance") -> plt.Figure:
        df = importance_df.head(top_n).copy()
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#D4A843" if i == 0 else "#4A5568" for i in range(len(df))]
        ax.barh(range(len(df)), df["importance"].values, color=colors[::-1])
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["feature"].values[::-1])
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        return fig

    @staticmethod
    def local_waterfall(waterfall_data: Dict, title: str = "Local Explanation") -> plt.Figure:
        features = waterfall_data["features"]
        names = [f["name"] for f in features]
        contributions = [f["shap_value"] for f in features]
        colors = ["#E53E3E" if c > 0 else "#38A169" for c in contributions]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(names)), contributions, color=colors[::-1])
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1])
        ax.set_xlabel("SHAP Value (impact on risk score)")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        return fig

    @staticmethod
    def summary_beeswarm(shap_values: np.ndarray, X: np.ndarray,
                          feature_names: List[str],
                          title: str = "SHAP Summary") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          show=False, plot_size=(10, 6))
        plt.title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig
