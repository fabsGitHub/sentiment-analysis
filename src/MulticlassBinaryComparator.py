import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import clone
import copy

from config import StrategyName, VectorizerName, TargetCol, enforce_reproducibility

class MulticlassBinaryComparator:
    def __init__(self, df: pd.DataFrame, label_encoder: LabelEncoder, strategy: StrategyName = StrategyName.MASKED, random_state: int = 42):
        self.df = df.copy()
        self.label_encoder = label_encoder
        self.strategy = strategy.value if hasattr(strategy, 'value') else str(strategy)
        self.random_state = random_state
        self.all_errors = []
        self.performance_metrics = []
        enforce_reproducibility(self.random_state)

    def run_comparative_matrix(self, models_grid: dict, vectorizers: list, ngram_ranges: list, samplers: dict, task3_results_df: pd.DataFrame = None):
        if self.strategy not in self.df.columns:
            print(f" [!] Configuration Error: Preprocessing column '{self.strategy}' missing.")
            return

        target_col = TargetCol.SENTIMENT.value

        if task3_results_df is not None:
            for _, row in task3_results_df.iterrows():
                self.performance_metrics.append({
                    "Task Setting": "Multiclass",
                    "Model": row["Model"],
                    "Vectorizer": row["Vectorizer"],
                    "N-Gram": row["N-Gram"],
                    "Sampling Strategy": row["Sampling Strategy"],
                    "Macro-F1": float(row["Macro-F1"]),
                    "Hyperparameters": row["Sampled_Hyperparameters"]
                })

        # Find what integer index 'neutral' maps to
        neutral_idx = -1
        for idx, cls_name in enumerate(self.label_encoder.classes_):
            if str(cls_name).lower() == 'neutral':
                neutral_idx = idx
                break

        if neutral_idx == -1:
            print(" [!] Warning: Could not explicitly locate 'neutral' class. Binary task might be corrupted.")
            df_binary = self.df.copy()
        else:
            # Filter rows where the target is NOT neutral
            df_binary = self.df[self.df[target_col] != neutral_idx].copy()

        df_clean = df_binary.dropna(subset=[self.strategy, target_col]).reset_index(drop=True)

        # Re-encode binary classes to 0 and 1 so models don't get confused by missing class indices
        binary_le = LabelEncoder()
        y_binary_encoded = binary_le.fit_transform(df_clean[target_col])

        # Train/Test Split on completely distinct Binary sets
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            df_clean[self.strategy], y_binary_encoded,
            test_size=0.20, random_state=self.random_state, stratify=y_binary_encoded
        )

        for model_enum, model_cfg in models_grid.items():
            model_str = model_enum.value
            model_cls = model_cfg["class"]

            for vec_enum in vectorizers:
                vec_str = vec_enum.value
                for ngram_tuple in ngram_ranges:

                    # Instantiate distinct vectorized pipeline layers
                    if vec_str == VectorizerName.BOW.value:
                        vect = CountVectorizer(ngram_range=ngram_tuple, max_features=5000)
                    else:
                        vect = TfidfVectorizer(ngram_range=ngram_tuple, max_features=5000)

                    X_train_vec = vect.fit_transform(X_train_raw)
                    X_test_vec = vect.transform(X_test_raw)

                    for sampler_enum, sampler_instance in samplers.items():
                        sampler_str = sampler_enum.value

                        try:
                            # Apply resampling
                            if sampler_instance is not None:
                                X_tr_res, y_tr_res = clone(sampler_instance).fit_resample(X_train_vec, y_train)
                            else:
                                X_tr_res, y_tr_res = X_train_vec, y_train

                            # Sample basic fallback hyperparameter map block
                            sampled_params = {}
                            for p_name, p_vals in model_cfg["param_grid"].items():
                                sampled_params[p_name] = p_vals[0] # select baseline validation state

                            clf = model_cls(**sampled_params)
                            clf.fit(X_tr_res, y_tr_res)
                            y_pred = clf.predict(X_test_vec)

                            macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

                            # APPEND TRULY UNIQUE METRICS
                            self.performance_metrics.append({
                                "Task Setting": "Binary (No Neutral)",
                                "Model": model_str,
                                "Vectorizer": vec_str,
                                "N-Gram": str(ngram_tuple),
                                "Sampling Strategy": sampler_str,
                                "Macro-F1": float(macro_f1),
                                "Hyperparameters": str(sampled_params)
                            })

                        except Exception as e:
                            continue
