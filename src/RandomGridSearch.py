import os
import random
import ast
import pandas as pd
import numpy as np
import joblib

# RAPIDS GPU-Beschleunigung (Auf Kaggle standardmäßig vorinstalliert)
try:
    import cupy as cp
    import cuml
    from cuml.linear_model import LogisticRegression as cuML_LogisticRegression
    from cuml.ensemble import RandomForestClassifier as cuML_RandomForest
    from cuml.svm import SVC as cuML_SVC
    from cuml.naive_bayes import MultinomialNB as cuML_MNB
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone

class RandomGridSearch:
    """
    Maximierte GPU-Pipeline-Suche.
    Verlagert das Training traditioneller ML-Modelle komplett auf die GPU via RAPIDS cuML
    und fängt I/O-Flaschenhälse durch GPU-In-Memory-Arrays ab.
    """
    def __init__(self, df: pd.DataFrame, target_col: str = "sentiment", random_state: int = 42, cache_dir: str = "data/matrix_cache"):
        self.df = df.copy()
        self.target_col = target_col
        self.random_state = random_state
        self.results_df = pd.DataFrame()
        self.cache_dir = cache_dir

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        if not GPU_AVAILABLE:
            print(" [!] WARNUNG: RAPIDS cuML nicht gefunden. Fallback auf langsame CPU-Modelle!")

    def _sample_parameters(self, param_grid: dict) -> dict:
        return {param: random.choice(values) if isinstance(values, list) else values
                for param, values in param_grid.items()}

    def _convert_to_gpu_model(self, base_model_cls):
        """Maps standard scikit-learn classes to ultra-fast cuML GPU classes."""
        if not GPU_AVAILABLE:
            return base_model_cls

        classname = base_model_cls.__name__
        if "LogisticRegression" in classname:
            return cuML_LogisticRegression
        elif "RandomForest" in classname:
            return cuML_RandomForest
        elif "SVC" in classname:
            return cuML_SVC
        elif "MultinomialNB" in classname:
            return cuML_MNB
        return base_model_cls  # Fallback für PyTorchMLPClassifier, der bereits GPU nutzt

    def fit(self,
            models_grid: dict,
            preprocessors: list,
            vectorizers: list,
            ngram_ranges: list,
            samplers: dict,
            n_iter_per_hyperparam: int = 1) -> pd.DataFrame:

        raw_results = []
        custom_token_pattern = r'(?u)\[?\b\w[-\w\.]*\b\]?'

        # Mapping der Modelle auf GPU-Pendants vorab
        prepared_models = [
            (str(k.value if hasattr(k, 'value') else k),
             self._convert_to_gpu_model(meta["class"]),
             meta["param_grid"])
            for k, meta in models_grid.items()
        ]

        for strategy in preprocessors:
            strategy_col = strategy.value if hasattr(strategy, 'value') else str(strategy)

            df_clean = self.df.dropna(subset=[strategy_col, self.target_col]).reset_index(drop=True)
            if df_clean.empty: continue

            try:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    df_clean[strategy_col], df_clean[self.target_col],
                    test_size=0.20, random_state=self.random_state, stratify=df_clean[self.target_col]
                )
            except ValueError:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    df_clean[strategy_col], df_clean[self.target_col],
                    test_size=0.20, random_state=self.random_state
                )

            for vec_type in vectorizers:
                vec_str = vec_type.value if hasattr(vec_type, 'value') else str(vec_type)

                for ngram in ngram_ranges:
                    ngram_tuple = ngram if isinstance(ngram, tuple) else ast.literal_eval(str(ngram))

                    vec_cache_prefix = f"{strategy_col}_{vec_str}_ngram_{ngram_tuple[0]}_{ngram_tuple[1]}"
                    x_test_cache_path = os.path.join(self.cache_dir, f"{vec_cache_prefix}_X_test.pkl")

                    # Vektorisierung (Immer noch CPU-gebunden, wird aber durch joblib abgefangen)
                    if os.path.exists(x_test_cache_path):
                        X_test_vec = joblib.load(x_test_cache_path)
                        X_train_vec = None
                    else:
                        vect = (CountVectorizer(ngram_range=ngram_tuple, token_pattern=custom_token_pattern, max_features=5000)
                                if vec_str == "BoW" else
                                TfidfVectorizer(ngram_range=ngram_tuple, token_pattern=custom_token_pattern, max_features=5000))

                        X_train_vec = vect.fit_transform(X_train_raw)
                        X_test_vec = vect.transform(X_test_raw)
                        joblib.dump(X_test_vec, x_test_cache_path)

                    for sampler_key, sampler_obj in samplers.items():
                        sampler_str = sampler_key.value if hasattr(sampler_key, 'value') else str(sampler_key)

                        data_state_suffix = f"{vec_cache_prefix}_sampler_{sampler_str}.pkl"
                        train_features_path = os.path.join(self.cache_dir, f"X_train_{data_state_suffix}")
                        train_labels_path = os.path.join(self.cache_dir, f"y_train_{data_state_suffix}")

                        if os.path.exists(train_features_path) and os.path.exists(train_labels_path):
                            X_train_res = joblib.load(train_features_path)
                            y_train_res = joblib.load(train_labels_path)
                        else:
                            if X_train_vec is None:
                                vect = (CountVectorizer(ngram_range=ngram_tuple, token_pattern=custom_token_pattern, max_features=5000)
                                        if vec_str == "BoW" else
                                        TfidfVectorizer(ngram_range=ngram_tuple, token_pattern=custom_token_pattern, max_features=5000))
                                X_train_vec = vect.fit_transform(X_train_raw)

                            if sampler_obj is not None:
                                try:
                                    X_train_res, y_train_res = clone(sampler_obj).fit_resample(X_train_vec, y_train)
                                except Exception:
                                    X_train_res, y_train_res = X_train_vec, y_train
                            else:
                                X_train_res, y_train_res = X_train_vec, y_train

                            joblib.dump(X_train_res, train_features_path)
                            joblib.dump(y_train_res, train_labels_path)

                        # --- STRUKTUR-OPTIMIERUNG: KONVERTIERUNG IN GPU-ARRAYS ---
                        # Wir schieben die CPU-Matrizen genau einmal hier auf die GPU.
                        # Dadurch entfällt das Hin- und Herschieben in der inneren Hyperparameter-Schleife.
                        if GPU_AVAILABLE:
                            try:
                                # Konvertiert scipy sparse Matrizen in cuML-kompatible GPU-bestückte CSR-Matrizen
                                X_train_gpu = cuml.common.input_utils.sparse_scipy_to_cp(X_train_res, dtype=np.float32)
                                X_test_gpu = cuml.common.input_utils.sparse_scipy_to_cp(X_test_vec, dtype=np.float32)
                                y_train_gpu = cp.asarray(y_train_res.values if hasattr(y_train_res, 'values') else y_train_res)
                            except Exception:
                                # Fallback, falls Konvertierung fehlschlägt
                                X_train_gpu, X_test_gpu, y_train_gpu = X_train_res, X_test_vec, y_train_res
                        else:
                            X_train_gpu, X_test_gpu, y_train_gpu = X_train_res, X_test_vec, y_train_res

                        for model_str, base_model_cls, hyperparam_grid in prepared_models:
                            print(f" [->] GPU-Exploring: Model={model_str} | Strategy={strategy_col} | Vectorizer={vec_str} | N-Gram={ngram_tuple} | Sampler={sampler_str}")

                            for _ in range(n_iter_per_hyperparam):
                                sampled_params = self._sample_parameters(hyperparam_grid)

                                # Säubere Parameter, die cuML nicht mag (z.B. n_jobs, da GPU inhärent parallel ist)
                                if GPU_AVAILABLE and "cuml" in base_model_cls.__module__:
                                    sampled_params.pop('n_jobs', None)

                                try:
                                    clf = base_model_cls(**sampled_params)

                                    # Fit und Predict laufen jetzt mit Lichtgeschwindigkeit auf der GPU
                                    clf.fit(X_train_gpu, y_train_gpu)
                                    y_pred_gpu = clf.predict(X_test_gpu)

                                    # Zurückholen auf CPU nur für die Metrik-Berechnung
                                    if hasattr(y_pred_gpu, "get"):
                                        y_pred = y_pred_gpu.get()
                                    elif isinstance(y_pred_gpu, cp.ndarray):
                                        y_pred = cp.asnumpy(y_pred_gpu)
                                    else:
                                        y_pred = y_pred_gpu

                                    raw_results.append({
                                        "Model": model_str,
                                        "Strategy": strategy_col,
                                        "Vectorizer": vec_str,
                                        "N-Gram": str(ngram_tuple),
                                        "Sampling Strategy": sampler_str,
                                        "Accuracy": float(accuracy_score(y_test, y_pred)),
                                        "Macro-F1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                                        "Sampled_Hyperparameters": str(sampled_params)
                                    })
                                except Exception as e:
                                    print(f" [!] Fehler beim GPU-Fitting von {model_str}: {str(e)}")
                                    continue

        self.results_df = pd.DataFrame(raw_results) if raw_results else pd.DataFrame()
        return self.results_df
