import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import optuna

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.utils.multiclass import type_of_target
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import issparse


class ClassificationHyperTuner:
    def __init__(self, model_name: str, X_train, y_train, X_test=None, y_test=None, n_trials=100, use_cv=False, use_smote=False):

        self.check_target_type(y_train)
        if use_smote:
            X_train, y_train = self.get_smote(X_train, y_train)

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
            "cat": (self.boost_cat, cb.CatBoostClassifier)
        }
        
    def check_target_type(self, y_train):    
        target_type = type_of_target(y_train)
        if target_type == 'binary':
            self.problem_type = 'binary'
            self.num_classes = None
        elif target_type == 'multiclass':
            self.problem_type = 'multiclass'
            self.num_classes = len(set(y_train))
        else:
            raise ValueError(f"Tipo de target não suportado: {target_type}")
        
    def get_smote(self, X_train, y_train):
        print(f"Original class distribution: {Counter(y_train)}")
        strategy = 'auto' if self.problem_type == 'binary' else 'not majority'
        
        # Verifica se X_train é uma matriz esparsa e converte
        if issparse(X_train):
            X_train_dense = X_train.toarray()
        else:
            X_train_dense = X_train

        smote = SMOTE(sampling_strategy=strategy, random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_dense, y_train)
        
        print(f"Resampled class distribution: {Counter(y_train_res)}")

        X_train = X_train_res
        y_train = y_train_res
        return X_train, y_train
        

    def get_average(self):
        return 'binary' if self.problem_type == 'binary' else 'macro'

    def evaluate_model(self, model):
        model.fit(self.X_train, self.y_train)
        predict = model.predict(self.X_test)
        return f1_score(self.y_test, predict, average=self.get_average())

    def objective(self, trial):
        try:
            if self.model_name not in self.model_map:
                raise ValueError(f"Modelo '{self.model_name}' não suportado.")

            param_func, model_class = self.model_map[self.model_name]
            params = param_func(trial)
            model = model_class(**params)

            if self.use_cv:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scorer = make_scorer(f1_score, average=self.get_average())
                scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scorer)#, n_jobs=-1)
                return scores.mean()
            else:
                return self.evaluate_model(model)

        except Exception as e:
            print(f"[Erro Trial {trial.number}] {e}")
            return float("nan")

    def run_optimization(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

        _, model_class = self.model_map[self.model_name]
        model = model_class(**study.best_params)
        model.fit(self.X_train, self.y_train)

        return study.best_params, study.best_value

    def boost_cat(self, trial):
        common_params = {
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
            "verbose": 0
        }

        if self.problem_type == "binary":
            common_params.update({
                "loss_function": "Logloss",
                "eval_metric": "AUC"
            })
        else:
            common_params.update({
                "loss_function": "MultiClass",
                "eval_metric": "Accuracy"
            })
        return common_params

    def boost_xgb(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 500, 1200 if learning_rate < 0.03 else 1000)
        common_params = {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            # "lambda": trial.suggest_float("lambda", 1e-3, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),

            "alpha": trial.suggest_float("alpha", 1e-3, 2.0, log=True),
            "random_state": 42,
            "verbosity": 0
        }

        if self.problem_type == "binary":
            common_params["objective"] = "binary:logistic"
            common_params["eval_metric"] = "logloss"
        else:
            common_params["objective"] = "multi:softprob"
            common_params["eval_metric"] = "mlogloss"
            common_params["num_class"] = self.num_classes

        return common_params

    def boost_lgb(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 500, 2500 if learning_rate < 0.03 else 1500)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        common_params = {
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
            "class_weight": class_weight,
            "verbose": -1,
            "random_state": 42
        }

        if self.problem_type == "binary":
            common_params["objective"] = "binary"
            common_params["metric"] = "binary_logloss"
        else:
            common_params["objective"] = "multiclass"
            common_params["metric"] = "multi_logloss"
            common_params["num_class"] = self.num_classes

        return common_params