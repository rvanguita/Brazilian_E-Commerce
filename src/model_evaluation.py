from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve
from sklearn.calibration import calibration_curve
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, matthews_corrcoef, cohen_kappa_score, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, brier_score_loss
    
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import shap



class ClassificationEvaluator:
    def __init__(self, model):
        self.model = model


    def _validate_inputs(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = np.asarray(y).ravel()
        return X, y


    def _calculate_metrics(self, y_true, y_pred, y_proba):
        n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 1
        is_multiclass = n_classes > 2

        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'Matthews Corrcoef': matthews_corrcoef(y_true, y_pred),
            'Cohen Kappa': cohen_kappa_score(y_true, y_pred),
        }

        # Brier Score
        if is_multiclass:
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes), sparse_output=False)
            brier = np.mean(np.array([
                brier_score_loss(y_true_bin[:, i], y_proba[:, i])
                for i in range(n_classes)
            ]))
        else:
            # para binário, y_proba é 1D ou a segunda coluna da matriz
            y_proba_bin = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            brier = brier_score_loss(y_true, y_proba_bin)
        
        metrics['Brier Score'] = brier
        metrics['Log Loss'] = log_loss(y_true, y_proba)

        # ROC AUC
        try:
            if is_multiclass:
                metrics['ROC AUC'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            else:
                metrics['ROC AUC'] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        except:
            metrics['ROC AUC'] = np.nan


        return {k: round(v * 100, 2) if k not in ['Matthews Corrcoef', 'Cohen Kappa'] else round(v, 2)
                for k, v in metrics.items()}


    def fit_evaluate(self, X, y, predict_proba=True):
        X, y = self._validate_inputs(X, y)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        n_classes = len(np.unique(y))

        if predict_proba and hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X)

            if n_classes == 2 and y_proba.shape[1] == 2:
                y_proba = y_proba 
        else:
            y_proba = np.zeros((len(y), n_classes), dtype=float)

        metrics = self._calculate_metrics(y, y_pred, y_proba)
        return pd.DataFrame([metrics]), X, y, y_pred, y_proba


    def cross_validate(self, X, y, n_splits=5, predict_proba=True):
        X, y = self._validate_inputs(X, y)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        classes = np.unique(y)
        n_classes = len(classes)

        metrics_agg = {k: [] for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC',
                                       'Matthews Corrcoef', 'Cohen Kappa', 'Brier Score', 'Log Loss']}
        X_test_all, y_true_all, y_pred_all, y_proba_all = [], [], [], []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            if predict_proba and hasattr(self.model, "predict_proba"):
                y_proba = self.model.predict_proba(X_test)
            else:
                y_proba = np.zeros((len(X_test), n_classes), dtype=float)
                
            X_test_all.append(X_test.to_numpy())
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_proba_all.append(y_proba)

            metrics = self._calculate_metrics(y_test, y_pred, y_proba)
            for k in metrics_agg:
                metrics_agg[k].append(metrics[k])

        final_metrics = pd.DataFrame([{k: round(np.mean(v), 2) for k, v in metrics_agg.items()}])

        return (
            final_metrics,
            np.vstack(X_test_all),
            np.array(y_true_all),
            np.array(y_pred_all),
            np.vstack(y_proba_all)
        )
        
    
    def cross_val_predict_summary(self, X, y, cv=5):
        X, y = self._validate_inputs(X, y)
        y_pred = cross_val_predict(self.model, X, y, cv=cv, method='predict')

        n_classes = len(np.unique(y))

        if hasattr(self.model, "predict_proba"):
            y_proba = cross_val_predict(self.model, X, y, cv=cv, method='predict_proba')

            if n_classes == 2 and y_proba.shape[1] == 2:
                y_proba = y_proba  
        else:
            y_proba = np.zeros((len(y), n_classes), dtype=float)

        metrics = self._calculate_metrics(y, y_pred, y_proba)
        return pd.DataFrame([metrics]), X, y, y_pred, y_proba
    
    


class ClassificationPlotter:
    def __init__(self, true_labels, predicted_labels, predicted_probabilities,
                 features=None, model=None, figsize=(12, 8)):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.predicted_probabilities = predicted_probabilities
        self.features = features
        self.model = model
        self.figsize = figsize

    def plot_confusion_matrix(self, ax, cmap=sns.cubehelix_palette(as_cmap=True)):
        
        classes = unique_labels(self.true_labels, self.predicted_labels)

        cm = confusion_matrix(self.true_labels, self.predicted_labels, normalize='true', labels=classes)
        raw_cm = confusion_matrix(self.true_labels, self.predicted_labels, labels=classes)

        # Texto: % + quantidade bruta
        labels = np.array([
            f"{pct:.1%}\n({raw})" for raw, pct in zip(raw_cm.flatten(), cm.flatten())
        ]).reshape(cm.shape)

        sns.heatmap(
            cm,
            annot=labels,
            fmt='',
            ax=ax,
            cmap=cmap,
            cbar=True,
            xticklabels=classes,
            yticklabels=classes,
            linewidths=0.5,
            linecolor='gray',
            annot_kws={"fontsize": 10, "ha": "center", "va": "center"}
        )

        ax.set_title('Confusion Matrix', fontsize=14, weight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


    def plot_roc_curve(self, ax, color='#c53b53'):
        if len(np.unique(self.true_labels)) == 2:
            fpr, tpr, _ = roc_curve(self.true_labels, self.predicted_probabilities)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=3, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.legend(loc="lower right")
        else:
            ax.text(0.5, 0.5, "ROC for multiclass not supported here", ha='center', va='center', fontsize=12)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontsize=16, weight='bold')
        for spine in ax.spines.values():
            spine.set_visible(False)


    def plot_precision_recall_curve(self, ax):
        if len(np.unique(self.true_labels)) == 2:
            precision, recall, _ = precision_recall_curve(self.true_labels, self.predicted_probabilities)
            ax.plot(recall, precision, lw=2)
        else:
            ax.text(0.5, 0.5, "ROC for multiclass not supported here", ha='center', va='center', fontsize=12)
            
            
        ax.set_title('Precision-Recall Curve', fontsize=16, weight='bold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        for spine in ax.spines.values():
            spine.set_visible(False)


    def plot_probability_distribution(self, ax=None):
        proba = np.array(self.predicted_probabilities)
        n_classes = proba.shape[1] if proba.ndim > 1 else 1

        if n_classes == 1:
            # Binary classification case
            sns.histplot(
                proba,
                bins=20,
                kde=True,
                ax=ax,
                stat="density"
            )
            ax.set_title("Probability Distribution", fontsize=16, weight='bold')
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Frequency")
        else:
            palette = sns.color_palette("tab10", n_classes)

            for i in range(n_classes):
                sns.kdeplot(
                    proba[:, i],
                    ax=ax,
                    label=f'Class {i}',
                    # fill=True,             
                    alpha=0.3,             
                    # common_norm=False,
                    # stat="density",
                    linewidth=2,           
                    color=palette[i]
                )

            ax.set_title("Probability Distribution by Class", fontsize=14, weight='bold')
            ax.set_xlabel("Predicted Probability", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.legend(title="Classes", title_fontsize=11, fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4) 

        for spine in ax.spines.values():
            spine.set_visible(False)

    
    
    def plot_calibration_curve(self, ax, n_bins=10):
        n_classes = len(np.unique(self.true_labels))
        proba = np.array(self.predicted_probabilities)

        if proba.ndim == 1:
            # Corrige caso tenha vindo 1D para binário
            proba = np.column_stack([1 - proba, proba])

        if n_classes > 2:
            y_true_bin = label_binarize(self.true_labels, classes=np.unique(self.true_labels))

            for i in range(n_classes):
                if i >= proba.shape[1]:
                    print(f"Aviso: probabilidade para a classe {i} não está presente. Ignorando.")
                    continue

                prob_true, prob_pred = calibration_curve(
                    y_true_bin[:, i], proba[:, i],
                    n_bins=n_bins, strategy='uniform'
                )
                ax.plot(prob_pred, prob_true, marker='o', label=f'Classe {i}')
        else:
            if proba.shape[1] < 2:
                raise ValueError("Esperado pelo menos duas colunas em predicted_probabilities para problema binário.")
            prob_true, prob_pred = calibration_curve(
                self.true_labels, proba[:, 1],
                n_bins=n_bins, strategy='uniform'
            )
            ax.plot(prob_pred, prob_true, marker='o', label='Calibration')

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        ax.set_title('Calibration Curve', fontsize=14, weight='bold')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.legend()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        for spine in ax.spines.values():
            spine.set_visible(False)


    def plot_learning_curve(self, ax, scoring='roc_auc', cv=5):
        if self.model is None:
            raise ValueError("Model must be provided for learning curve plot.")
        if not hasattr(self.model, 'fit') or not hasattr(self.model, 'score'):
            raise ValueError("Model must implement 'fit' and 'score' methods (e.g., sklearn Pipeline or estimator).")
        
        unique_labels = np.unique(self.true_labels)
        if len(unique_labels) > 2 and scoring == 'roc_auc':
            scoring = 'roc_auc_ovr'

        train_sizes, train_scores, val_scores = learning_curve(
            self.model, 
            self.features, 
            self.true_labels, 
            cv=cv, 
            scoring=scoring,
            n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)
        
        ax.plot(train_sizes, train_mean, label="Training score")
        ax.plot(train_sizes, val_mean, label="Cross-validation score")
        ax.set_title("Learning Curve")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(scoring.capitalize())
        ax.legend()



    def plot_selected_charts(self, charts):
        chart_methods = {
            'confusion_matrix': self.plot_confusion_matrix,
            'roc_curve': self.plot_roc_curve,
            'precision_recall_curve': self.plot_precision_recall_curve,
            'probability_distribution': self.plot_probability_distribution,
            'calibration_curve': self.plot_calibration_curve,
            'learning_curve': self.plot_learning_curve,
        }

        wide_charts = {'learning_curve', 'probability_distribution'}  # charts that take 2 columns
        selected = sorted(
            [chart for chart in charts if chart in chart_methods],
            key=lambda x: x in wide_charts
        )

        width_units = 0
        cols = 2
        for chart in selected:
            span = 2 if chart in wide_charts else 1
            if width_units % cols + span > cols:
                width_units += cols - (width_units % cols) 
            width_units += span

        rows = (width_units + cols - 1) // cols

        fig = plt.figure(figsize=(cols * self.figsize[0], rows * self.figsize[1]))
        gs = gridspec.GridSpec(rows, cols, figure=fig)
        current_cell = 0

        for chart in selected:
            method = chart_methods[chart]
            span = 2 if chart in wide_charts else 1

            if current_cell % cols + span > cols:
                current_cell += cols - (current_cell % cols)  # move to next row

            row, col = divmod(current_cell, cols)

            if row >= rows:
                raise IndexError(f"GridSpec index ({row}) is out of bounds for {rows} rows.")

            if span == 2:
                ax = fig.add_subplot(gs[row, :])
            else:
                ax = fig.add_subplot(gs[row, col])

            method(ax)
            current_cell += span

        plt.tight_layout()
        plt.show()
        
        
        