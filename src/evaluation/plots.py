import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from tqdm import tqdm
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
from typing import Any, List, Optional, Tuple, Union


class ClassificationVisualizer:
    def __init__(
        self,
        true_labels: Union[List, np.ndarray],
        predicted_labels: Union[List, np.ndarray],
        predicted_probabilities: Union[List, np.ndarray],
        features: Union[pd.DataFrame, np.ndarray, None] = None,
        model: Optional[Any] = None,
        figsize: Tuple[int, int] = (12, 8),
    ):
        self.true_labels = np.array(true_labels)
        self.predicted_labels = np.array(predicted_labels)
        self.predicted_probabilities = np.array(predicted_probabilities)
        self.features = features
        self.model = model
        self.figsize = figsize

    def _binarize_labels(self) -> np.ndarray:
        return label_binarize(self.true_labels, classes=np.unique(self.true_labels))

    def _validate_probabilities(self) -> np.ndarray:
        if np.any(self.predicted_probabilities < 0) or np.any(self.predicted_probabilities > 1):
            raise ValueError("Probabilities must be in the range [0, 1].")

        if self.predicted_probabilities.ndim == 2 and self.predicted_probabilities.shape[1] == 2:
            return self.predicted_probabilities[:, 1]

        return self.predicted_probabilities

    def _style_axis(self, ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
        ax.set_title(title, fontsize=18, fontweight='bold') if title else None
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold') if xlabel else None
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold') if ylabel else None
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def plot_confusion_matrix(self, ax: plt.Axes, cmap: Any = sns.cubehelix_palette(as_cmap=True)) -> None:
        classes = unique_labels(self.true_labels, self.predicted_labels)
        normalized_cm = confusion_matrix(self.true_labels, self.predicted_labels, normalize='true', labels=classes)
        raw_cm = confusion_matrix(self.true_labels, self.predicted_labels, labels=classes)
        labels = np.array([
            f"{pct:.1%}\n({raw})" for raw, pct in zip(raw_cm.flatten(), normalized_cm.flatten())
        ]).reshape(normalized_cm.shape)

        sns.heatmap(
            normalized_cm,
            annot=labels,
            fmt='',
            ax=ax,
            cmap=cmap,
            cbar=True,
            xticklabels=classes,
            yticklabels=classes,
            linewidths=0.5,
            linecolor='gray',
            annot_kws={"fontsize": 16, "ha": "center", "va": "center"}
        )
        self._style_axis(ax, "Confusion Matrix", "Predicted", "True")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
    def plot_roc_curve(self, ax: plt.Axes, color: str = '#c53b53') -> None:
        probabilities = self._validate_probabilities()
        num_classes = len(np.unique(self.true_labels))

        if num_classes == 2:
            fpr, tpr, _ = roc_curve(self.true_labels, probabilities)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=3, label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            y_bin = self._binarize_labels()
            for i in range(num_classes):
                if i >= probabilities.shape[1]:
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2.5, label=f'Class {i} (AUC = {roc_auc:.2f})')

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)
        self._style_axis(ax, "ROC Curve", "False Positive Rate", "True Positive Rate")
        ax.legend(fontsize=12)

    def plot_precision_recall_curve(self, ax: plt.Axes) -> None:
        probabilities = self._validate_probabilities()
        num_classes = len(np.unique(self.true_labels))

        if num_classes == 2:
            precision, recall, _ = precision_recall_curve(self.true_labels, probabilities)
            ax.plot(recall, precision, lw=2.5)
        else:
            y_bin = self._binarize_labels()
            for i in range(num_classes):
                if i >= probabilities.shape[1]:
                    continue
                precision, recall, _ = precision_recall_curve(y_bin[:, i], probabilities[:, i])
                ax.plot(recall, precision, lw=2.5, label=f'Class {i}')

        self._style_axis(ax, f'Precision-Recall Curve ({num_classes}-class)', "Recall", "Precision")
        ax.legend(fontsize=12)

    def plot_probability_distribution(self, ax: Optional[plt.Axes] = None) -> None:
        probabilities = self.predicted_probabilities
        num_classes = probabilities.shape[1] if probabilities.ndim > 1 else 1

        if num_classes == 1:
            sns.histplot(probabilities, bins=20, kde=True, ax=ax, stat="density")
            self._style_axis(ax, "Probability Distribution", "Predicted Probability", "Density")
        else:
            palette = sns.color_palette("tab10", num_classes)
            for i in range(num_classes):
                sns.kdeplot(probabilities[:, i], ax=ax, label=f'Class {i}', linewidth=2, color=palette[i])
            self._style_axis(ax, "Probability Distribution by Class", "Predicted Probability", "Density")
            ax.legend(title="Classes", title_fontsize=11, fontsize=12)

    def plot_calibration_curve(self, ax: plt.Axes, n_bins: int = 10) -> None:
        num_classes = len(np.unique(self.true_labels))
        probabilities = self.predicted_probabilities

        if probabilities.ndim == 1:
            probabilities = np.column_stack([1 - probabilities, probabilities])

        if num_classes > 2:
            y_true_bin = self._binarize_labels()
            for i in range(num_classes):
                if i >= probabilities.shape[1]:
                    continue
                prob_true, prob_pred = calibration_curve(
                    y_true_bin[:, i], probabilities[:, i], n_bins=n_bins, strategy='uniform'
                )
                ax.plot(prob_pred, prob_true, marker='o', linestyle='-', linewidth=2.5, label=f'Class {i}')
        else:
            prob_true, prob_pred = calibration_curve(
                self.true_labels, probabilities[:, 1], n_bins=n_bins, strategy='uniform'
            )
            ax.plot(prob_pred, prob_true, marker='o', linestyle='-', linewidth=2.5, label='Calibration')

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, alpha=0.6, label='Perfect Calibration')
        self._style_axis(ax, "Calibration Curve", "Mean Predicted Probability", "Fraction of Positives")
        ax.legend(fontsize=12)

    def plot_selected_charts(self, charts: List[str], cols: int = 2) -> None:
        chart_functions = {
            'confusion_matrix': self.plot_confusion_matrix,
            'roc_curve': self.plot_roc_curve,
            'precision_recall_curve': self.plot_precision_recall_curve,
            'probability_distribution': self.plot_probability_distribution,
            'calibration_curve': self.plot_calibration_curve,
        }

        wide_charts = {'probability_distribution'}
        selected = sorted(charts, key=lambda c: c in wide_charts)

        width_units = 0
        for chart in selected:
            span = 2 if chart in wide_charts else 1
            if width_units % cols + span > cols:
                width_units += cols - (width_units % cols)
            width_units += span

        rows = (width_units + cols - 1) // cols
        fig = plt.figure(figsize=(cols * self.figsize[0], rows * self.figsize[1]))
        grid_spec = gridspec.GridSpec(rows, cols, figure=fig)
        current_cell = 0

        for chart in tqdm(selected, desc="Plotting charts"):
            func = chart_functions[chart]
            span = 2 if chart in wide_charts else 1
            if current_cell % cols + span > cols:
                current_cell += cols - (current_cell % cols)
            row, col = divmod(current_cell, cols)
            ax = fig.add_subplot(grid_spec[row, :]) if span == 2 else fig.add_subplot(grid_spec[row, col])
            func(ax)
            current_cell += span

        plt.tight_layout()
        plt.show()