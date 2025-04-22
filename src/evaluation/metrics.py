from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, matthews_corrcoef, cohen_kappa_score, brier_score_loss
    
)
from sklearn.preprocessing import label_binarize

import pandas as pd
import numpy as np




class ClassifierMetricsEvaluator:
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
    
    