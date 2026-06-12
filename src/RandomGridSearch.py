import random
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone

class RandomGridSearch:
    """
    Executes an optimized hybrid structural pipeline exploration sequence.
    Hoists structural elements (Preprocessors, Vectorizers, Samplers) to the
    outermost loops to completely avoid redundant CPU computation.
    """
    def __init__(self, df: pd.DataFrame, target_col: str = "sentiment", random_state: int = 42):
        self.df = df.copy()
        self.target_col = target_col
        self.random_state = random_state
        self.results_df = pd.DataFrame()

        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

    def _sample_parameters(self, param_grid: dict) -> dict:
        """Helper to randomly extract configuration parameters from a distribution space."""
        sampled = {}
        for param, values in param_grid.items():
            if isinstance(values, list):
                sampled[param] = random.choice(values)
            else:
                sampled[param] = values
        return sampled

    def fit(self,
            models_grid: dict,
            preprocessors: list,
            vectorizers: list,
            ngram_ranges: list,
            samplers: dict,
            n_iter_per_hyperparam: int = 1) -> pd.DataFrame:
        """
        Executes an optimized algorithmic structure grid across workspace elements.
        """
        raw_results = []
        custom_token_pattern = r'(?u)\[?\b\w[-\w\.]*\b\]?'

        # LOOP HOISTING: Strategy stays on the outside
        for strategy in preprocessors:
            strategy_col = strategy.value if hasattr(strategy, 'value') else str(strategy)

            df_clean = self.df.dropna(subset=[strategy_col, self.target_col]).reset_index(drop=True)
            if df_clean.empty:
                print(f" [!] Warning: Clean dataframe is empty for strategy column: {strategy_col}")
                continue

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

            # LOOP HOISTING: Vectorization layers execute prior to downstream sampling/modeling
            for vec_type in vectorizers:
                vec_str = vec_type.value if hasattr(vec_type, 'value') else str(vec_type)

                for ngram in ngram_ranges:
                    ngram_tuple = ngram if isinstance(ngram, tuple) else ast.literal_eval(str(ngram))

                    if vec_str == "BoW":
                        vect = CountVectorizer(ngram_range=ngram_tuple, token_pattern=custom_token_pattern, max_features=5000)
                    else:
                        vect = TfidfVectorizer(ngram_range=ngram_tuple, token_pattern=custom_token_pattern, max_features=5000)

                    try:
                        X_train_vec = vect.fit_transform(X_train_raw)
                        X_test_vec = vect.transform(X_test_raw)
                    except Exception as ve:
                        print(f" [!] Vectorizer error on {vec_str} {ngram_tuple}: {str(ve)}")
                        continue

                    # LOOP HOISTING: Resampling occurs once per vectorized array state
                    for sampler_key, sampler_obj in samplers.items():
                        sampler_str = sampler_key.value if hasattr(sampler_key, 'value') else str(sampler_key)

                        if sampler_obj is not None:
                            try:
                                X_train_res, y_train_res = clone(sampler_obj).fit_resample(X_train_vec, y_train)
                            except Exception:
                                X_train_res, y_train_res = X_train_vec, y_train
                        else:
                            X_train_res, y_train_res = X_train_vec, y_train

                        # Downstream processing loops digest fully built datasets directly
                        for model_key, model_meta in models_grid.items():
                            base_model_cls = model_meta["class"]
                            hyperparam_grid = model_meta["param_grid"]
                            model_str = model_key.value if hasattr(model_key, 'value') else str(model_key)

                            print(f" [->] Exploring: Model={model_str} | Strategy={strategy_col} | Vectorizer={vec_str} | N-Gram={ngram} | Sampler={sampler_str}")

                            for _ in range(n_iter_per_hyperparam):
                                sampled_params = self._sample_parameters(hyperparam_grid)

                                try:
                                    clf = base_model_cls(**sampled_params)
                                    clf.fit(X_train_res, y_train_res)
                                    y_pred = clf.predict(X_test_vec)

                                    acc = accuracy_score(y_test, y_pred)
                                    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

                                    raw_results.append({
                                        "Model": str(model_str),
                                        "Strategy": str(strategy_col),
                                        "Vectorizer": str(vec_str),
                                        "N-Gram": str(ngram_tuple),
                                        "Sampling Strategy": str(sampler_str),
                                        "Accuracy": float(acc),
                                        "Macro-F1": float(macro_f1),
                                        "Sampled_Hyperparameters": str(sampled_params)
                                    })
                                except Exception as e:
                                    print(f" [!] Error fitting classifier {model_str}: {str(e)}")
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
