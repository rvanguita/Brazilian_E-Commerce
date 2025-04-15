import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import shap
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import optuna

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    log_loss, matthews_corrcoef, cohen_kappa_score, roc_curve, auc, confusion_matrix
)


class ShapPlot:
    def __init__(self, model, X_test, df_feature, features_drop=None):
        self.model = model
        self.X_test = X_test
        
        self.feature_names = df_feature.columns
        if features_drop:
            self.feature_names = df_feature.drop(features_drop, axis=1).columns
            
        self.explainer = shap.Explainer(self.model)  # Reutiliza o explainer para consistência
        
    def first_analysis(self, extension=False):
        shap_values = self.explainer(self.X_test)

        # Ajuste os nomes das features
        shap_values.feature_names = self.feature_names

        # Gráficos principais
        shap.plots.beeswarm(shap_values)
        shap.plots.bar(shap_values)
        shap.waterfall_plot(shap_values[0])

        # Extensão opcional para análise adicional
        if extension:
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names)
        
    def complete(self, comparison=False, analysis=None, interaction_index=None, show_interactions=False):
        # Calcule os valores SHAP
        shap_values = self.explainer(self.X_test)

        # Converter X_test para DataFrame para melhores visualizações
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)

        # Criar o objeto Explanation para o force_plot
        shap_explanation = shap.Explanation(
            values=shap_values.values[0],  # Os valores SHAP
            base_values=self.explainer.expected_value,  # O valor esperado (média das previsões)
            data=X_test_df.iloc[0],  # A amostra de entrada que estamos explicando
            feature_names=self.feature_names  # Os nomes das features
        )

        # Exibir força de explicação para uma previsão
        shap.initjs()
        shap.force_plot(shap_explanation.base_values, shap_explanation.values, shap_explanation.data)

        # Criar gráfico de decisão
        shap.decision_plot(shap_explanation.base_values, shap_explanation.values, shap_explanation.data)

        # Calcular e exibir interações SHAP, se necessário
        if show_interactions:
            shap_interaction_values = self.explainer.shap_interaction_values(self.X_test)  # Correção para `self.explainer`
            shap.summary_plot(shap_interaction_values, self.X_test, feature_names=self.feature_names)

        # Se `comparison` for verdadeiro, criar gráfico de dependência
        if comparison and analysis is not None:
            shap.dependence_plot(analysis, shap_values, self.X_test, feature_names=self.feature_names, interaction_index=interaction_index)


class ClassificationHyperTuner:
    def __init__(self, X_train, y_train, X_test, y_test, n_trials=100, model_name=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_trials = n_trials
        self.model_name = model_name

    def objective(self, trial):
        try:
            if self.model_name == "lgb":
                params = self.boost_lgb(trial)
                model = lgb.LGBMClassifier(**params)
            elif self.model_name == "xgb":
                params = self.boost_xgb(trial)
                model = xgb.XGBClassifier(**params)
            elif self.model_name == "cat":
                params = self.boost_cat(trial)
                model = cb.CatBoostClassifier(**params)

            model.fit(self.X_train, self.y_train)
            preds = model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, preds)
            return auc

        except Exception as e:
            print(f"[Erro Trial] {e}")
            return float("nan")

    def run_optimization(self):
        # Criação do estudo Optuna
        study = optuna.create_study(direction='maximize')
        
        # Execução da otimização
        study.optimize(self.objective, n_trials=self.n_trials)

        # Verificação do melhor resultado
        # print(f"Best hyperparameters: {study.best_params}")
        # print(f"Best AUC: {study.best_value}")

        return study.best_params, study.best_value

    def train_and_evaluate(self, model):
        # Fit the model and evaluate it on the test set
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict_proba(self.X_test)[:, 1]
        # accuracy = accuracy_score(self.y_test, predictions > 0.5)
        
        auc = roc_auc_score(self.y_test, predictions)
        return auc

    def boost_cat(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.05, log=True)

        params = {
            "iterations": trial.suggest_int(
                "iterations", 
                500, 
                2000 if learning_rate < 0.05 else 1000
            ),
            "learning_rate": learning_rate,
            "depth": trial.suggest_int("depth", 4, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 20.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
            "border_count": trial.suggest_int("border_count", 128, 300),
            "od_type": trial.suggest_categorical("od_type", ["Iter", "IncToDec"]),
            "random_seed": 42,
            "verbose": 0
        }

        return params
    
    def boost_xgb(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)  # Ajustado para uma faixa mais realista

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "learning_rate": learning_rate,
            "n_estimators": trial.suggest_int(
                "n_estimators", 
                500, 
                1200 if learning_rate < 0.03 else 1000  # Faixa ajustada
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 6),  # Faixa ajustada para uma profundidade mais controlada
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),  # Faixa ajustada
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),  # Faixa expandida para mais flexibilidade
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  # Faixa expandida
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 2.0, log=True),  # Faixa ajustada
            "alpha": trial.suggest_float("alpha", 1e-3, 2.0, log=True),  # Faixa ajustada
            "random_state": 42,
            "verbosity": 0
        }
        
        return params

    def boost_lgb(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": learning_rate,
            "n_estimators": trial.suggest_int(
                "n_estimators", 
                500, 
                2500 if learning_rate < 0.03 else 1500
            ),
            "num_leaves": trial.suggest_int("num_leaves", 31, 128),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.7, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "max_bin": trial.suggest_int("max_bin", 200, 255),
            "verbose": -1,
            "random_state": 42
        }

        return params


class TrainingValidation:
    def __init__(self, model, rouc_curve=True, confusion_matrix=True, figsize=(12, 5)):
        self.rouc_curve = rouc_curve
        self.model = model
        self.confusion_matrix = confusion_matrix
        self.figsize =figsize


    def plot_roc_curve(self, fpr, tpr, roc_auc, ax=None, color='#c53b53'):
        if ax is None:
            ax = plt.gca()
            
        # plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color=color, lw=3, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Linha diagonal
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve', fontsize=16, weight='bold')
        plt.legend(loc="lower right")
        
        ax.grid(False)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

        plt.show()


    def calculate_metrics(self, y, predictions, predictions_proba):
        metrics = {
            'Accuracy': accuracy_score(y, predictions),
            'Precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'Recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'F1 Score': f1_score(y, predictions, average='weighted', zero_division=0),
            'ROC AUC': roc_auc_score(y, predictions_proba),
            'Matthews Corrcoef': matthews_corrcoef(y, predictions),
            'Cohen Kappa': cohen_kappa_score(y, predictions),
            'Log Loss': log_loss(y, predictions_proba)
        }
        return {k: round(v * 100, 2) if k != 'Matthews Corrcoef' and k != 'Cohen Kappa' else round(v, 2) 
                for k, v in metrics.items()}


    def plot_confusion_matrix(self, y, predictions, ax=None):
        if ax is None:
            ax = plt.gca()
            
        cm = confusion_matrix(y, predictions)
        labels = np.asarray(
            [
                ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
                for item in cm.flatten()
            ]
        ).reshape(2, 2)

        sns.heatmap(cm, annot=labels, fmt="", ax=ax, cbar=False)
        ax.set_title('Confusion Matrix', fontsize=14, weight='bold')
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

        

    def normal(self, X, y, oversampling=False):
        if oversampling:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)  
        
        
        self.model.fit(X, y)
        predictions_proba = self.model.predict_proba(X)[:, 1]
        predictions = self.model.predict(X)
        
        scores = self.calculate_metrics(y, predictions, predictions_proba)
        scores_df = pd.DataFrame([scores])
                  
        if self.confusion_matrix and self.rouc_curve:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            self.plot_confusion_matrix(y, predictions, ax=ax1)
            fpr, tpr, _ = roc_curve(y, predictions_proba)
            roc_auc = auc(fpr, tpr)
            self.plot_roc_curve(fpr, tpr, roc_auc, ax=ax2)
            plt.tight_layout()
            plt.show()

        elif self.confusion_matrix:
            self.plot_confusion_matrix(y, predictions)

        elif self.rouc_curve:
            fpr, tpr, _ = roc_curve(y, predictions_proba)
            roc_auc = auc(fpr, tpr)
            self.plot_roc_curve(fpr, tpr, roc_auc)

        return scores_df


    def cross(self, X, y, n_splits=5, oversampling=False):
        # Garantir que X seja DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Garantir que y seja array 1D
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0].values.ravel()
        elif isinstance(y, pd.Series):
            y = y.values.ravel()
        elif hasattr(y, "ravel"):
            y = y.ravel()

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        metrics_cross = {
            'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': [],
            'Matthews Corrcoef': [], 'Cohen Kappa': [], 'Log Loss': []
        }

        all_predictions = []
        all_true_labels = []
        all_predictions_proba = []

        for idx_train, idx_test in cv.split(X, y):
            X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]

            if oversampling:
                smote = SMOTE()
                X_train, y_train = smote.fit_resample(X_train, y_train)

            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            predict_proba = self.model.predict_proba(X_test)[:, 1]

            metrics = self.calculate_metrics(y_test, predictions, predict_proba)
            for key in metrics_cross:
                metrics_cross[key].append(metrics[key])

            # Acumulando predições e verdadeiros
            all_predictions.extend(predictions)
            all_true_labels.extend(y_test)
            all_predictions_proba.extend(predict_proba)

        if self.confusion_matrix and self.rouc_curve:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            self.plot_confusion_matrix(np.array(all_true_labels), np.array(all_predictions), ax=ax1)
            fpr, tpr, _ = roc_curve(all_true_labels, all_predictions_proba)
            roc_auc = auc(fpr, tpr)
            self.plot_roc_curve(fpr, tpr, roc_auc, ax=ax2)
            plt.tight_layout()
            plt.show()
        elif self.confusion_matrix:
            self.plot_confusion_matrix(np.array(all_true_labels), np.array(all_predictions))
        elif self.rouc_curve:
            fpr, tpr, _ = roc_curve(all_true_labels, all_predictions_proba)
            roc_auc = auc(fpr, tpr)
            self.plot_roc_curve(fpr, tpr, roc_auc)

        # Agregando médias das métricas
        scores = {key: round(np.mean(val), 2) for key, val in metrics_cross.items()}
        return pd.DataFrame([scores])



    def cross_val_(self, X_train, y_train, cv=5):
        metrics_cross = {key: [] for key in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 
                                             'Matthews Corrcoef', 'Cohen Kappa', 'Log Loss']}
        
        y_pred_train = cross_val_predict(self.model, X_train, y_train, cv=cv, method='predict')
        y_proba_train = cross_val_predict(self.model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
        
        metrics = self.calculate_metrics(y_train, y_pred_train, y_proba_train)
        for key in metrics_cross.keys():
            metrics_cross[key].append(metrics[key])
        scores = {key: round(np.mean(val), 2) for key, val in metrics.items()}
        
        scores_df = pd.DataFrame([scores])
        
        if self.confusion_matrix:
            self.plot_confusion_matrix(y_train, y_pred_train)


        if self.rouc_curve:
            fpr, tpr, _ = roc_curve(y_train, y_proba_train)
            roc_auc = auc(fpr, tpr)
            self.plot_roc_curve(fpr, tpr, roc_auc)
        
        return scores_df