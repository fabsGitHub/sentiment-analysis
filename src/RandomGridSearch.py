import os
import random
import ast
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone

class RandomGridSearch:
    """
    Strukturell optimierte Pipeline-Suche.
    Garantiert minimale CPU-Auslastung durch strikte Hierarchie-Ebenen,
    In-Memory RAM-Sicherung und bereinigte Schleifen-Prozesse.
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

    def _sample_parameters(self, param_grid: dict) -> dict:
        return {param: random.choice(values) if isinstance(values, list) else values
                for param, values in param_grid.items()}

    def fit(self,
            models_grid: dict,
            preprocessors: list,
            vectorizers: list,
            ngram_ranges: list,
            samplers: dict,
            n_iter_per_hyperparam: int = 1) -> pd.DataFrame:

        raw_results = []
        custom_token_pattern = r'(?u)\[?\b\w[-\w\.]*\b\]?'

        # Pre-Caching von String-Repräsentationen der Modelle, um CPU-Overhead in Schleifen zu minimieren
        prepared_models = [
            (str(k.value if hasattr(k, 'value') else k), meta["class"], meta["param_grid"])
            for k, meta in models_grid.items()
        ]

        for strategy in preprocessors:
            strategy_col = strategy.value if hasattr(strategy, 'value') else str(strategy)

            df_clean = self.df.dropna(subset=[strategy_col, self.target_col]).reset_index(drop=True)
            if df_clean.empty:
                print(f" [!] Warnung: DataFrame leer für Strategie: {strategy_col}")
                continue

            # Train-Test-Split stabil und einmalig pro Text-Pipeline extrahieren
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

                    # --- STRUKTUR-OPTIMIERUNG 1: Test-Vektoren RAM-Zuweisung ---
                    if os.path.exists(x_test_cache_path):
                        X_test_vec = joblib.load(x_test_cache_path)
                        X_train_vec = None  # Wird nur bei Bedarf geladen/berechnet
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

                        # --- STRUKTUR-OPTIMIERUNG 2: Speicher- & RAM-Schonung bei Resampling ---
                        if os.path.exists(train_features_path) and os.path.exists(train_labels_path):
                            X_train_res = joblib.load(train_features_path)
                            y_train_res = joblib.load(train_labels_path)
                        else:
                            # Falls X_train_vec noch nicht berechnet wurde (weil X_test aus dem Cache kam)
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

                        # --- STRUKTUR-OPTIMIERUNG 3: Blitzschnelle Classifier-Schleife ---
                        # X_train_res und y_train_res bleiben für ALLE Modelle fest im RAM verankert!
                        for model_str, base_model_cls, hyperparam_grid in prepared_models:
                            print(f" [->] Exploring: Model={model_str} | Strategy={strategy_col} | Vectorizer={vec_str} | N-Gram={ngram_tuple} | Sampler={sampler_str}")

                            for _ in range(n_iter_per_hyperparam):
                                sampled_params = self._sample_parameters(hyperparam_grid)

                                try:
                                    clf = base_model_cls(**sampled_params)
                                    clf.fit(X_train_res, y_train_res)
                                    y_pred = clf.predict(X_test_vec)

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
                                    print(f" [!] Fehler beim Fitten von {model_str}: {str(e)}")
                                    continue

        if not raw_results:
            print(" [!] CRITICAL: Grid search complete but ZERO pipeline variations executed successfully.")
            self.results_df = pd.DataFrame(columns=[
                "Model", "Strategy", "Vectorizer", "N-Gram", "Sampling Strategy", "Accuracy", "Macro-F1", "Sampled_Hyperparameters"
            ])
        else:
            self.results_df = pd.DataFrame(raw_results)
            print(f" [+] Grid search successfully populated {len(self.results_df)} logging rows.")

        return self.results_df
