import logging
from collections import Counter
from typing import Any, Callable, Dict, Optional, Tuple, Union

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HyperparameterTuner:
    def __init__(
        self,
        features: Any,
        labels: Any,
        pipeline: Pipeline,
        test_size: float = 0.2,
        n_trials: int = 100,
        use_cross_validation: bool = False,
        model_step: str = "model"
    ):
        self.n_trials = n_trials
        self.use_cv = use_cross_validation
        self.pipeline = pipeline
        self.model_step = model_step

        if self.model_step not in self.pipeline.named_steps:
            raise ValueError(f"Pipeline must include a step named '{self.model_step}'.")

        self._detect_problem_type(labels)
        self._split_data(features, labels, test_size)

        self.param_generators: Dict[Any, Callable] = {
            lgb.LGBMClassifier: self._lgb_params,
            xgb.XGBClassifier: self._xgb_params,
            cb.CatBoostClassifier: self._catboost_params,
        }

        self._assign_param_generator()

    def _assign_param_generator(self) -> None:
        model = self.pipeline.named_steps.get(self.model_step)
        for model_class, param_func in self.param_generators.items():
            if isinstance(model, model_class):
                self.param_generator = param_func
                return
        raise ValueError(f"Unsupported model type: {type(model).__name__}")

    def _detect_problem_type(self, labels: Any) -> None:
        target_type = type_of_target(labels)
        if target_type == "binary":
            self.problem_type = "binary"
            self.num_classes = 2
        elif target_type == "multiclass":
            self.problem_type = "multiclass"
            self.num_classes = len(set(labels))
        else:
            raise ValueError(f"Unsupported target type: {target_type}")

    def _split_data(self, X: Any, y: Any, test_size: float) -> None:
        if self.use_cv:
            self.X_train, self.y_train = X, y
            self.X_test = self.y_test = None
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )

    def _get_average_strategy(self) -> str:
        if self.problem_type == "binary":
            return "binary"
        class_distribution = Counter(self.y_train)
        imbalance_ratio = max(class_distribution.values()) / min(class_distribution.values())
        return "weighted" if imbalance_ratio > 1.5 else "macro"

    def _get_eval_metric(self) -> Optional[str]:
        model = self.pipeline.named_steps[self.model_step]
        if isinstance(model, xgb.XGBClassifier):
            return "auc" if self.problem_type == "binary" else "mlogloss"
        elif isinstance(model, lgb.LGBMClassifier):
            return "auc" if self.problem_type == "binary" else "multi_logloss"
        elif isinstance(model, cb.CatBoostClassifier):
            return "AUC" if self.problem_type == "binary" else "TotalF1"
        return None

    def _objective(self, trial: optuna.Trial) -> float:
        try:
            params = self.param_generator(trial)
            trial_pipeline = clone(self.pipeline)
            trial_pipeline.set_params(**{f"{self.model_step}__{k}": v for k, v in params.items()})
            return self._cross_val_score(trial_pipeline) if self.use_cv else self._evaluate(trial_pipeline)
        except Exception as e:
            logger.warning(f"[Trial failed] {e}")
            return float("nan")

    def _cross_val_score(self, model: Pipeline) -> float:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring_func = make_scorer(f1_score, average=self._get_average_strategy())
        scores = []

        for train_idx, val_idx in kf.split(self.X_train, self.y_train):
            X_train, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train, y_val = self.y_train[train_idx], self.y_train[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(scoring_func._score_func(y_val, y_pred, average=self._get_average_strategy()))
        return float(np.mean(scores))

    def _evaluate(self, model: Pipeline) -> float:
        model.fit(self.X_train, self.y_train)
        if self.X_test is not None:
            preds = model.predict(self.X_test)
            return f1_score(self.y_test, preds, average=self._get_average_strategy())
        return model.score(self.X_train, self.y_train)

    def _catboost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
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
            "eval_metric": self._get_eval_metric(),
            "loss_function": "Logloss" if self.problem_type == "binary" else "MultiClass"
        }

    def _xgb_params(self, trial: optuna.Trial) -> Dict[str, Any]:
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
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 2.0, log=True),
            "random_state": 42,
            "verbosity": 0,
            "eval_metric": self._get_eval_metric(),
            "objective": "binary:logistic" if self.problem_type == "binary" else "multi:softprob"
        }
        if self.problem_type == "multiclass":
            params["num_class"] = self.num_classes
        return params

    def _lgb_params(self, trial: optuna.Trial) -> Dict[str, Any]:
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
            "metric": self._get_eval_metric(),
            "objective": "binary" if self.problem_type == "binary" else "multiclass"
        }
        if self.problem_type == "multiclass":
            params["num_class"] = self.num_classes
        return params

    def run(self) -> Tuple[Dict[str, Any], float]:
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)

        final_pipeline = clone(self.pipeline)
        final_pipeline.set_params(**{f"{self.model_step}__{k}": v for k, v in study.best_params.items()})
        final_pipeline.fit(self.X_train, self.y_train)

        if self.X_test is not None:
            self._evaluate(final_pipeline)

        return study.best_params, study.best_value