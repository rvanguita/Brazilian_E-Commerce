import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import optuna

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score


class ClassificationHyperTuner:
    def __init__(self, X_train, y_train, X_test, y_test, model_name: str, n_trials=100, use_cv=False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_trials = n_trials
        self.model_name = model_name
        self.use_cv = use_cv

        self.model_map = {
            "lgb": (self.boost_lgb, lgb.LGBMClassifier),
            "xgb": (self.boost_xgb, xgb.XGBClassifier),
            "cat": (self.boost_cat, cb.CatBoostClassifier),
            "tree_dt": (self.boost_tree_dt, DecisionTreeClassifier),
            "rf": (self.boost_rf, RandomForestClassifier)
        }

    def objective(self, trial):
        try:
            if self.model_name not in self.model_map:
                raise ValueError(f"Modelo '{self.model_name}' não suportado.")

            param_func, model_class = self.model_map[self.model_name]
            params = param_func(trial)
            model = model_class(**params)

            if self.use_cv:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring="roc_auc")
                return scores.mean()
            else:
                model.fit(self.X_train, self.y_train)
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(self.X_test)[:, 1]
                else:
                    preds = model.decision_function(self.X_test)
                auc = roc_auc_score(self.y_test, preds)
                return auc

        except Exception as e:
            print(f"[Erro Trial] {e}")
            return float("nan")

    def run_optimization(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Treinar melhor modelo com os melhores hiperparâmetros
        _, model_class = self.model_map[self.model_name]
        model = model_class(**study.best_params)
        model.fit(self.X_train, self.y_train)

        return study.best_params, study.best_value

    def train_and_evaluate(self, model):
        model.fit(self.X_train, self.y_train)
        preds = model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, preds)
        return auc

    def boost_cat(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.05, log=True)
        return {
            "iterations": learning_rate,
            "learning_rate": learning_rate,
            "depth": trial.suggest_int("depth", 4, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 20.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
            "border_count": trial.suggest_int("border_count", 128, 300),
            "od_type": trial.suggest_categorical("od_type", ["Iter", "IncToDec"]),
            "loss_function": trial.suggest_categorical("loss_function", ["Logloss", "CrossEntropy"]),
            "eval_metric": trial.suggest_categorical("eval_metric", ["AUC", "Accuracy", "Logloss"]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),


            "random_seed": 42,
            "verbose": 0
        }
    
    def boost_xgb(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 500, 1200 if learning_rate < 0.03 else 1000)

        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": trial.suggest_int("max_depth", 3, 6),  # Faixa ajustada para uma profundidade mais controlada
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),  # Faixa ajustada
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),  # Faixa expandida para mais flexibilidade
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  # Faixa expandida
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 2.0, log=True),  # Faixa ajustada
            "alpha": trial.suggest_float("alpha", 1e-3, 2.0, log=True),  # Faixa ajustada
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0),

            "random_state": 42,
            "verbosity": 0
        }

    def boost_lgb(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 500, 2500 if learning_rate < 0.03 else 1500)

        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "num_leaves": trial.suggest_int("num_leaves", 31, 128),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.7, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "max_bin": trial.suggest_int("max_bin", 200, 255),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0),
            "verbose": -1,
            "random_state": 42
        }

    def boost_tree_dt(self, trial):
        return {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 2, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 100),


            "random_state": 42
        }
    
    def boost_rf(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": bootstrap,
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),

            "random_state": 42
        }