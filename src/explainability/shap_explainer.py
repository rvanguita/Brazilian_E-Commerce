import shap
import pandas as pd
from functools import lru_cache

class ShapExplainer:
    def __init__(self, model, X_test, df_feature, features_drop=None, verbose=False):
        """
        Classe para análise e visualização de explicações SHAP.
        """
        self.model = model
        self.X_test = X_test
        self.verbose = verbose

        # Verificação e definição dos nomes das features
        self.feature_names = df_feature.columns
        if features_drop:
            self.feature_names = df_feature.drop(features_drop, axis=1).columns
        assert X_test.shape[1] == len(self.feature_names), "Mismatch entre X_test e df_feature"

        # Criar o explicador uma única vez
        self.explainer = shap.Explainer(self.model)

    @lru_cache()
    def _shap_values(self):
        """Cache dos valores SHAP para evitar múltiplos cálculos."""
        if self.verbose:
            print("Calculando valores SHAP...")
        return self.explainer(self.X_test)

    def plot_basic_summary(self, extension=False):
        """Plota gráficos beeswarm, bar e waterfall com SHAP."""
        shap_values = self._shap_values()
        shap_values.feature_names = self.feature_names

        shap.plots.beeswarm(shap_values)
        shap.plots.bar(shap_values)
        shap.plots.waterfall(shap_values[0])

        if extension:
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names)

    def plot_force_plot(self, sample_idx=0):
        """Plota gráfico de força para uma amostra específica."""
        shap_values = self._shap_values()
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)

        shap_explanation = shap.Explanation(
            values=shap_values.values[sample_idx],
            base_values=self.explainer.expected_value,
            data=X_test_df.iloc[sample_idx],
            feature_names=self.feature_names
        )

        shap.initjs()
        shap.force_plot(shap_explanation.base_values, shap_explanation.values, shap_explanation.data)

    def plot_detailed_report(self, sample_idx=0, comparison=False, analysis=None, interaction_index=None, show_interactions=False):
        """Plota relatório completo de análise SHAP para uma amostra."""
        shap_values = self._shap_values()
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)

        # Force plot e decision plot
        shap_explanation = shap.Explanation(
            values=shap_values.values[sample_idx],
            base_values=self.explainer.expected_value,
            data=X_test_df.iloc[sample_idx],
            feature_names=self.feature_names
        )
        shap.initjs()
        shap.force_plot(shap_explanation.base_values, shap_explanation.values, shap_explanation.data)
        shap.decision_plot(shap_explanation.base_values, shap_explanation.values, shap_explanation.data)

        if show_interactions:
            shap_interaction_values = self.explainer.shap_interaction_values(self.X_test)
            shap.summary_plot(shap_interaction_values, self.X_test, feature_names=self.feature_names)

        if comparison and analysis is not None:
            shap.dependence_plot(analysis, shap_values, self.X_test, feature_names=self.feature_names, interaction_index=interaction_index)
