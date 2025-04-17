import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    log_loss, matthews_corrcoef, cohen_kappa_score, roc_curve, auc, confusion_matrix
)

    
class ClassificationValidator:
    def __init__(self, model, oversample=None, plot_roc=True, plot_cm=True, figsize=(12, 5)):
        self.model = model
        self.oversample = oversample
        self.plot_roc = plot_roc
        self.plot_cm = plot_cm
        self.figsize = figsize


    def _validate_inputs(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        y = np.asarray(y).ravel()
        
        return X, y


    def _calculate_metrics(self, y_true, y_pred, y_proba):
        if len(np.unique(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, y_proba)
            logloss = log_loss(y_true, y_proba)
            
        else:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
            logloss = log_loss(y_true, y_proba)

        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'ROC AUC': roc_auc,
            'Matthews Corrcoef': matthews_corrcoef(y_true, y_pred),
            'Cohen Kappa': cohen_kappa_score(y_true, y_pred),
            'Log Loss': logloss
        }
        
        return {k: round(v * 100, 2) if k not in ['Matthews Corrcoef', 'Cohen Kappa'] else round(v, 2)
                for k, v in metrics.items()}


    def _plot_confusion_matrix(self, y_true, y_pred, ax=None, normalize='true', cmap=sns.cubehelix_palette(as_cmap=True)):
        if ax is None:
            ax = plt.gca()

        cm = confusion_matrix(y_true, y_pred, normalize=normalize)

        # Exibe valores absolutos + percentual
        raw_cm = confusion_matrix(y_true, y_pred)
        labels = np.array([
            f"{raw}\n{pct:.1%}" for raw, pct in zip(raw_cm.flatten(), cm.flatten())
        ]).reshape(cm.shape)

        sns.heatmap(cm, annot=labels, fmt='', ax=ax, cmap=cmap, cbar=True, linewidths=0.5, linecolor='gray')

        ax.set_title('Confusion Matrix', fontsize=14, weight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')


    def _plot_roc_curve(self, y_true, y_proba, ax=None, color='#c53b53'):
        if ax is None:
            ax = plt.gca()
            
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
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


    def _plot_results(self, y_true, y_pred, y_proba):
        if self.plot_cm or self.plot_roc:
            fig, axes = plt.subplots(1, 2 if self.plot_cm and self.plot_roc else 1, figsize=self.figsize)
            axes = axes if isinstance(axes, np.ndarray) else [axes]
            
            if self.plot_cm:
                self._plot_confusion_matrix(y_true, y_pred, ax=axes[0])
                
            if self.plot_roc:
                self._plot_roc_curve(y_true, y_proba, ax=axes[-1])
                
            plt.tight_layout()
            plt.show()


    def fit_evaluate(self, X, y, oversample=False, predict_proba=True):
        X, y = self._validate_inputs(X, y)
        
        if oversample and self.oversample:
            X, y = self.oversample.fit_resample(X, y)

        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        if predict_proba:
            y_proba = self.model.predict_proba(X)
            y_proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
            
        else:
            y_proba = np.zeros_like(y, dtype=float)

        metrics = self._calculate_metrics(y, y_pred, y_proba)
        self._plot_results(y, y_pred, y_proba)

        return pd.DataFrame([metrics])


    def cross_validate(self, X, y, n_splits=5, oversample=False, predict_proba=True):
        X, y = self._validate_inputs(X, y)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        metrics_agg = {k: [] for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC',
                                       'Matthews Corrcoef', 'Cohen Kappa', 'Log Loss']}

        y_true_all, y_pred_all, y_proba_all = [], [], []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if oversample and self.oversample:
                X_train, y_train = self.oversample.fit_resample(X_train, y_train)

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            if predict_proba:
                y_proba = self.model.predict_proba(X_test)
                y_proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
            else:
                y_proba = np.zeros_like(y_pred, dtype=float)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_proba_all.extend(y_proba if y_proba.ndim == 1 else y_proba.max(axis=1))

            metrics = self._calculate_metrics(y_test, y_pred, y_proba)
            for k in metrics_agg:
                metrics_agg[k].append(metrics[k])

        self._plot_results(y_true_all, y_pred_all, y_proba_all)

        return pd.DataFrame([{k: round(np.mean(v), 2) for k, v in metrics_agg.items()}])


    def cross_val_predict_summary(self, X, y, cv=5):
        X, y = self._validate_inputs(X, y)
        y_pred = cross_val_predict(self.model, X, y, cv=cv, method='predict')
        y_proba = cross_val_predict(self.model, X, y, cv=cv, method='predict_proba')
        y_proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba

        metrics = self._calculate_metrics(y, y_pred, y_proba)
        self._plot_results(y, y_pred, y_proba)

        return pd.DataFrame([metrics])
    