import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

from FFNN import PyTorchMLPClassifier

class SentimentExperimentPipeline:
    """
    Orchestrates cross-combinatorial NLP experiments evaluating text preprocessing strategies,
    n-gram boundaries, balancing algorithms, and vectorization arrays against predictive backbones.
    """

    def __init__(self, data_path: str = "Sentences_50Agree.txt", target_col: str = "sentiment"):
        self.data_path = data_path
        self.target_col = target_col
        self.df: pd.DataFrame = pd.DataFrame()
        self.results_df: pd.DataFrame = pd.DataFrame()
        self.label_encoder = LabelEncoder()

        # Combinatorial Grid Configurations
        self.ngram_ranges = [(1, 1), (1, 2), (1, 3), (1, 4)]
        self.strategies = ["prep_standard", "prep_full", "prep_masked", "prep_standard_numbers"]
        self.custom_token_pattern = r'(?u)\[?\b\w[-\w\.]*\b\]?'

        # Setup Default Evaluated Framework Components
        self.samplers = {
            "None (Imbalanced)": None,
            "SMOTE (Oversampling)": SMOTE(random_state=42),
            "SMOTETomek (Combined)": SMOTETomek(random_state=42),
            "Tomek (Undersampling)": TomekLinks()
        }

        self.models = {
            "Naive Bayes": MultinomialNB(),
            "FFNN (Updated)": PyTorchMLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=128,
                learning_rate_init=0.005,
                max_iter=50,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        }

    def load_and_initialize_data(self, preprocessor_instance=None) -> pd.DataFrame:
        """Loads dataset from file fallback or creates simulated dummy frames."""
        try:
            self.df = pd.read_csv(self.data_path, sep="@", header=None, names=["sentence", self.target_col])
            print(f"Successfully ingested dataset from: {self.data_path}")
        except FileNotFoundError:
            print(f"Warning: '{self.data_path}' not found. Generating synthetically populated dummy dataframe...")
            dummy_data = {
                "sentence": [
                    "Nokia signs 3-year deal with alternative eur10m networks.",
                    "The financial group corporate profit fell by 50pct down.",
                    "The company oyj expects sales growth over next period."
                ] * 50,
                self.target_col: ["positive", "negative", "neutral"] * 50
            }
            self.df = pd.DataFrame(dummy_data)

        # Encode Targets Safely
        self.df[self.target_col] = self.label_encoder.fit_transform(self.df[self.target_col])

        # Apply preprocessing steps if an operational engine context is passed
        if preprocessor_instance is not None:
            print("Applying text preprocessing routines to dataset...")
            self.df["sentence"] = self.df["sentence"].drop_duplicates(keep="first")
            self.df["prep_standard"] = self.df["sentence"].apply(preprocessor_instance.preprocess, strategy="standard")
            self.df["prep_full"] = self.df["sentence"].apply(preprocessor_instance.preprocess, strategy="full")
            self.df["prep_standard_numbers"] = self.df["sentence"].apply(preprocessor_instance.preprocess, strategy="standard_optimized")
            self.df['prep_masked'] = self.df['sentence'].apply(preprocessor_instance.preprocess, strategy="masked")

        self.df = self.df.drop_duplicates(subset=['prep_masked'], keep="first")
        return self.df

    def run_experiment_matrix(self) -> pd.DataFrame:
        """Executes full grid evaluation sequence across models, resamplers, and vectorizers."""
        if self.df.empty:
            raise ValueError("Dataframe context is empty. Please invoke load_and_initialize_data() before evaluating.")

        raw_results = []

        for strategy in self.strategies:
            print(f"---> Processing Strategy Alignment: {strategy.upper()}")
            df_clean = self.df.dropna(subset=[strategy, self.target_col]).reset_index(drop=True)

            X_train, X_test, y_train, y_test = train_test_split(
                df_clean[strategy], df_clean[self.target_col],
                test_size=0.20, random_state=42, stratify=df_clean[self.target_col]
            )

            for ngram in self.ngram_ranges:
                # Setup Extractors
                count_vect = CountVectorizer(ngram_range=ngram, token_pattern=self.custom_token_pattern)
                X_train_bow_raw = count_vect.fit_transform(X_train)
                X_test_bow = count_vect.transform(X_test)

                tfidf_vect = TfidfVectorizer(ngram_range=ngram, token_pattern=self.custom_token_pattern)
                X_train_tfidf_raw = tfidf_vect.fit_transform(X_train)
                X_test_tfidf = tfidf_vect.transform(X_test)

                for sampler_name, sampler_instance in self.samplers.items():
                    # Handle Balancing Pipeline Transformations Safely
                    if sampler_instance is not None:
                        current_sampler_bow = clone(sampler_instance)
                        current_sampler_tfidf = clone(sampler_instance)
                        X_train_bow, y_train_resampled_bow = current_sampler_bow.fit_resample(X_train_bow_raw, y_train)
                        X_train_tfidf, y_train_resampled_tfidf = current_sampler_tfidf.fit_resample(X_train_tfidf_raw, y_train)
                    else:
                        X_train_bow, y_train_resampled_bow = X_train_bow_raw, y_train
                        X_train_tfidf, y_train_resampled_tfidf = X_train_tfidf_raw, y_train

                    final_feature_size_bow = X_train_bow.shape[1]
                    final_feature_size_tfidf = X_train_tfidf.shape[1]

                    for model_name, original_model_instance in self.models.items():
                        # --- BoW Fit Sequence ---
                        model_instance = clone(original_model_instance)
                        model_instance.fit(X_train_bow, y_train_resampled_bow)
                        y_pred_bow = model_instance.predict(X_test_bow)
                        raw_results.append({
                            "Model": model_name, "Strategy": strategy, "Sampling Strategy": sampler_name,
                            "Vectorizer": "BoW", "N-Gram": str(ngram), "Feature Size": final_feature_size_bow,
                            "Accuracy": accuracy_score(y_test, y_pred_bow), "Macro-F1": f1_score(y_test, y_pred_bow, average='macro')
                        })

                        # --- TF-IDF Fit Sequence ---
                        model_instance = clone(original_model_instance)
                        model_instance.fit(X_train_tfidf, y_train_resampled_tfidf)
                        y_pred_tfidf = model_instance.predict(X_test_tfidf)
                        raw_results.append({
                            "Model": model_name, "Strategy": strategy, "Sampling Strategy": sampler_name,
                            "Vectorizer": "TF-IDF", "N-Gram": str(ngram), "Feature Size": final_feature_size_tfidf,
                            "Accuracy": accuracy_score(y_test, y_pred_tfidf), "Macro-F1": f1_score(y_test, y_pred_tfidf, average='macro')
                        })

        self.results_df = pd.DataFrame(raw_results)

        # Enforce Explicit Categorical Ordering onto Strategy Attributes
        self.results_df['Strategy'] = pd.Categorical(
            self.results_df['Strategy'],
            categories=self.strategies,
            ordered=True
        )
        return self.results_df

    def get_summary_matrix(self, metric: str = "Macro-F1") -> pd.DataFrame:
        """Returns structured pivot view tracking target metric variants."""
        if self.results_df.empty:
            raise ValueError("No calculated evaluations found. Please invoke run_experiment_matrix() first.")

        return pd.pivot_table(
            self.results_df, values=metric,
            index=['Model', 'Sampling Strategy', 'Vectorizer', 'N-Gram', 'Feature Size'],
            columns=['Strategy'],
            aggfunc='max'
        )

    def export_results(self, filename: str = "Master_performance_matrix.csv"):
        """Exports the raw calculated evaluation configurations to a separated CSV log file."""
        if self.results_df.empty:
            raise ValueError("No tracking data available to save. Execute run_experiment_matrix() first.")

        self.results_df.to_csv(filename, sep=";", index=False)
        print(f"Successfully exported raw execution summary logs to: {filename}")


# ==========================================
# TESTING STANDARD EXECUTION ROUTINE
# ==========================================
if __name__ == "__main__":
    # Mocking downstream dependency classes for local operational test checks
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neural_network import MLPClassifier as PyTorchMLPClassifier  # Interchangeable fallback representation

    # Instantiate and initiate pipeline processing
    pipeline = SentimentExperimentPipeline()

    # Passing None skips active execution transformation transformations natively for dry running
    pipeline.load_and_initialize_data(preprocessor_instance=None)

    # Run the underlying performance calculations
    pipeline.run_experiment_matrix()

    # Pull structured summary matrix
    big_matrix = pipeline.get_summary_matrix(metric="Macro-F1")

    print("\n========================= THE MASTER PERFORMANCE MATRIX (Macro-F1) =========================")
    print(big_matrix.round(4))
    print("============================================================================================")

    # Export metrics securely to disk
    pipeline.export_results("Master_performance_matrix.csv")
