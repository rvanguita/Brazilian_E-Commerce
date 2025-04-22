import shap
import pandas as pd
from functools import lru_cache
from typing import Optional, Union, List


class ShapExplainer:
    """
    Utility class for SHAP-based model interpretation and visualization.
    """

    def __init__(
        self,
        model,
        X_test,
        df_feature: pd.DataFrame,
        features_drop: Optional[Union[str, List[str]]] = None,
        verbose: bool = False
    ):
        self.model = model
        self.X_test = X_test
        self.verbose = verbose

        self.feature_names = df_feature.drop(columns=features_drop).columns if features_drop else df_feature.columns
        assert X_test.shape[1] == len(self.feature_names), "Shape mismatch between X_test and feature names"

        self.explainer = shap.Explainer(self.model)

    @lru_cache()
    def _shap_values(self):
        """Computes and caches SHAP values for the test set."""
        if self.verbose:
            print("Computing SHAP values...")
        return self.explainer(self.X_test)

    def plot_basic_summary(self, include_extended: bool = False) -> None:
        """Plots standard SHAP visualizations: beeswarm, bar, and waterfall."""
        shap_values = self._shap_values()
        shap_values.feature_names = self.feature_names

        shap.plots.beeswarm(shap_values)
        shap.plots.bar(shap_values)
        shap.plots.waterfall(shap_values[0])

        if include_extended:
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names)

    def plot_force_plot(self, sample_idx: int = 0) -> None:
        """Displays a force plot for a single prediction."""
        shap_values = self._shap_values()
        X_df = pd.DataFrame(self.X_test, columns=self.feature_names)

        explanation = shap.Explanation(
            values=shap_values.values[sample_idx],
            base_values=self.explainer.expected_value,
            data=X_df.iloc[sample_idx],
            feature_names=self.feature_names
        )

        shap.initjs()
        shap.force_plot(explanation.base_values, explanation.values, explanation.data)

    def plot_detailed_report(
        self,
        sample_idx: int = 0,
        comparison: bool = False,
        analysis: Optional[str] = None,
        interaction_index: Optional[Union[int, str]] = None,
        show_interactions: bool = False
    ) -> None:
        """Creates an interactive SHAP report for a given sample."""
        shap_values = self._shap_values()
        X_df = pd.DataFrame(self.X_test, columns=self.feature_names)

        explanation = shap.Explanation(
            values=shap_values.values[sample_idx],
            base_values=self.explainer.expected_value,
            data=X_df.iloc[sample_idx],
            feature_names=self.feature_names
        )

        shap.initjs()
        shap.force_plot(explanation.base_values, explanation.values, explanation.data)
        shap.decision_plot(explanation.base_values, explanation.values, explanation.data)

        if show_interactions:
            interaction_values = self.explainer.shap_interaction_values(self.X_test)
            shap.summary_plot(interaction_values, self.X_test, feature_names=self.feature_names)

        if comparison and analysis:
            shap.dependence_plot(
                feature=analysis,
                shap_values=shap_values,
                features=self.X_test,
                feature_names=self.feature_names,
                interaction_index=interaction_index
            )
