import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import shap
import optuna

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.utils.multiclass import type_of_target
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import issparse



class ClassificationHyperTuner:
    def __init__(self, model_name: str, X_train, y_train, X_test=None, y_test=None, n_trials=100, use_cv=False, use_smote=False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_trials = n_trials
        self.use_cv = use_cv
        self.model_name = model_name

        self._check_target_type(y_train)
        if use_smote:
            self.X_train, self.y_train = self._apply_smote(X_train, y_train)

        self.model_specs = {
            "lgb": (lgb.LGBMClassifier, self._params_lgb, "binary", "multiclass"),
            "xgb": (xgb.XGBClassifier, self._params_xgb, "binary:logistic", "multi:softprob"),
            "cat": (cb.CatBoostClassifier, self._params_cat, "Logloss", "MultiClass")
        }
        
    def _check_target_type(self, y_train):    
        target_type = type_of_target(y_train)
        if target_type == 'binary':
            self.problem_type = 'binary'
            self.num_classes = None
        elif target_type == 'multiclass':
            self.problem_type = 'multiclass'
            self.num_classes = len(set(y_train))
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        
    def _apply_smote(self, X, y):
        strategy = 'auto' if self.problem_type == 'binary' else 'not majority'
        X_dense = X.toarray() if issparse(X) else X
        smote = SMOTE(sampling_strategy=strategy, random_state=42)
        return smote.fit_resample(X_dense, y)
        
    def _average_strategy(self):
        if self.problem_type == 'binary':
            return 'binary'
        class_counts = Counter(self.y_train)
        imbalance = max(class_counts.values()) / min(class_counts.values())
        return 'weighted' if imbalance > 1.5 else 'macro'

    def _eval_metric(self):
        if self.model_name == "xgb":
            return "auc" if self.problem_type == "binary" else "mlogloss"
        elif self.model_name == "lgb":
            return "auc" if self.problem_type == "binary" else "multi_logloss"
        elif self.model_name == "cat":
            return "AUC" if self.problem_type == "binary" else "TotalF1"

    def _objective(self, trial):
        try:
            model_class, param_fn, *_ = self.model_specs[self.model_name]
            params = param_fn(trial)
            model = model_class(**params)
            return self._cross_val_score(model) if self.use_cv else self._evaluate_model(model)
        except Exception:
            return float("nan")

    def _cross_val_score(self, model):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average=self._average_strategy())
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scorer)
        return scores.mean()

    def _evaluate_model(self, model):
        model.fit(self.X_train, self.y_train)
        if self.X_test is not None and self.y_test is not None:
            preds = model.predict(self.X_test)
            return f1_score(self.y_test, preds, average=self._average_strategy())

    def _params_cat(self, trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.05, log=True),
            "depth": trial.suggest_int("depth", 4, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 20.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
            "border_count": trial.suggest_int("border_count", 128, 300),
            "od_type": trial.suggest_categorical("od_type", ["Iter", "IncToDec"]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": self._eval_metric(),
            "loss_function": "Logloss" if self.problem_type == "binary" else "MultiClass"
        }
        return params

    def _params_xgb(self, trial):
        lr = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 500, 1200 if lr < 0.03 else 1000)
        params = {
            "learning_rate": lr,
            "n_estimators": n_estimators,
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            # "lambda": trial.suggest_float("lambda", 1e-3, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),

            "alpha": trial.suggest_float("alpha", 1e-3, 2.0, log=True),
            "random_state": 42,
            "verbosity": 0,
            "eval_metric": self._eval_metric(),
            "objective": "binary:logistic" if self.problem_type == "binary" else "multi:softprob"
        }

        if self.problem_type == "multiclass":
            params["num_class"] = self.num_classes

        return params

    def _params_lgb(self, trial):
        lr = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 500, 4000 if lr < 0.01 else 2000)

        params = {
            "learning_rate": lr,
            "n_estimators": n_estimators,
            "num_leaves": trial.suggest_int("num_leaves", 31, 128),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.7, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "max_bin": trial.suggest_int("max_bin", 200, 512),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "verbose": -1,
            "random_state": 42,
            "metric": self._eval_metric(),
            "objective": "binary" if self.problem_type == "binary" else "multiclass"
        }
        if self.problem_type == "multiclass":
            params["num_class"] = self.num_classes
        return params
    
    def run_optimization(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)

        model_class, *_ = self.model_specs[self.model_name]
        best_model = model_class(**study.best_params)
        best_model.fit(self.X_train, self.y_train)

        if self.X_test is not None and self.y_test is not None:
            self._evaluate_model(best_model)

        return study.best_params, study.best_value
