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
    Executes an optimized hyperparameter and parametric pipeline exploration sequence.
    Randomly samples search configurations over preprocessing methods, vectorizer matrices,
    balancing samplers, and downstream predictive models.
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
        """Helper to randomly extract a continuous/discrete dictionary from a distribution space."""
        sampled = {}
        for param, values in param_grid.items():
            if isinstance(values, list):
                sampled[param] = random.choice(values)
            else:
                # Fallback directly for non-iterable scalar configurations
                sampled[param] = values
        return sampled

    def fit(self,
            models_grid: dict,
            preprocessors: list,
            vectorizers: list,
            ngram_ranges: list,
            samplers: dict,
            n_iter: int = 10,
            custom_token_pattern: str = r'(?u)\[?\b\w[-\w\.]*\b\]?') -> pd.DataFrame:
        """
        Runs random combinations across pipeline parameters and records model metric signatures.

        :param models_grid: Dict mapping 'Model Name' -> dict with {'class': ClassRef, 'param_grid': {}}
        :param preprocessors: List of strings matching DataFrame preprocessing column names.
        :param vectorizers: List of strings containing matching types, e.g., ['BoW', 'TF-IDF'].
        :param ngram_ranges: List of tuples specifying token context dimensions.
        :param samplers: Dict mapping 'Sampler Name' -> Class instance reference / None.
        :param n_iter: Maximum unique permutations checked during grid processing.
        """
        raw_results = []

        # Build strict parameter combination matrix to sample from uniformly
        pipeline_space = []
        for model_name, model_meta in models_grid.items():
            for prep in preprocessors:
                for vec in vectorizers:
                    for ngram in ngram_ranges:
                        for sampler_name, sampler_inst in samplers.items():
                            pipeline_space.append({
                                "model_name": model_name,
                                "model_class": model_meta["class"],
                                "param_grid": model_meta.get("param_grid", {}),
                                "strategy": prep,
                                "vec_type": vec,
                                "ngram": ngram,
                                "sampler_name": sampler_name,
                                "sampler_instance": sampler_inst
                            })

        # Safeguard iterations bounds against absolute framework ceiling size
        actual_iterations = min(n_iter, len(pipeline_space))
        sampled_pipelines = random.sample(pipeline_space, actual_iterations)

        print(f"======================================================================")
        print(f"INITIATING RANDOMIZED SEARCH GRID: {actual_iterations} / {len(pipeline_space)} PIPELINES")
        print(f"======================================================================")

        for idx, pipe in enumerate(sampled_pipelines, 1):
            strategy = pipe["strategy"]
            target_col = self.target_col

            if strategy not in self.df.columns:
                print(f"[!] Target text strategy sequence '{strategy}' missing from context frame. Skipping...")
                continue

            print(f"[{idx}/{actual_iterations}] Model: {pipe['model_name']} | Preprocessing: {strategy} | Features: {pipe['vec_type']} {pipe['ngram']}")

            # Clear data dropouts systematically across selected preprocessing domains
            df_clean = self.df.dropna(subset=[strategy, target_col]).reset_index(drop=True)

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                df_clean[strategy], df_clean[target_col],
                test_size=0.20,
                random_state=self.random_state,
                stratify=df_clean[target_col]
            )

            # Extract Features contextually
            if pipe["vec_type"] == "BoW":
                vectorizer_instance = CountVectorizer(ngram_range=pipe["ngram"], token_pattern=custom_token_pattern, max_features=5000)
            else:
                vectorizer_instance = TfidfVectorizer(ngram_range=pipe["ngram"], token_pattern=custom_token_pattern, max_features=5000)

            X_train_vec = vectorizer_instance.fit_transform(X_train_raw)
            X_test_vec = vectorizer_instance.transform(X_test_raw)

            # Apply Class Re-balancers
            if pipe["sampler_instance"] is not None:
                current_sampler = clone(pipe["sampler_instance"])
                X_train_res, y_train_res = current_sampler.fit_resample(X_train_vec, y_train)
            else:
                X_train_res, y_train_res = X_train_vec, y_train

            # Sample Hyperparameters for this structural target run
            sampled_params = self._sample_parameters(pipe["param_grid"])

            # Instantiate Model using selected hyperparameter state properties
            clf = pipe["model_class"](**sampled_params)

            try:
                clf.fit(X_train_res, y_train_res)
                y_pred = clf.predict(X_test_vec)

                acc = accuracy_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average="macro")

                # Store diagnostic tracking footprint tracking log record
                result_entry = {
                    "Model": pipe["model_name"],
                    "Strategy": strategy,
                    "Vectorizer": pipe["vec_type"],
                    "N-Gram": str(pipe["ngram"]),
                    "Sampling Strategy": pipe["sampler_name"],
                    "Accuracy": acc,
                    "Macro-F1": macro_f1,
                    "Sampled_Hyperparameters": str(sampled_params)
                }
                raw_results.append(result_entry)

            except Exception as e:
                print(f" [!] Execution error encountered inside model processing pipeline iteration step: {str(e)}")
                continue

        self.results_df = pd.DataFrame(raw_results)
        return self.results_df

    def get_summary_matrix(self, metric: str = "Macro-F1") -> pd.DataFrame:
        """Pivots the collected raw logging metrics dataframe into a readable evaluation chart."""
        if self.results_df.empty:
            raise ValueError("Execution logs context is empty. Run search processing pipeline first.")

        return self.results_df.pivot_table(
            index=["Model", "Sampling Strategy", "Sampled_Hyperparameters"],
            columns=["Strategy", "Vectorizer", "N-Gram"],
            values=metric,
            aggfunc="max"
        )
