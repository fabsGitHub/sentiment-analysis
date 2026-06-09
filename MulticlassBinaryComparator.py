import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

from FFNN import PyTorchMLPClassifier

print("\n" + "="*70)
print("EXECUTING GENERALIZED MULTICLASS VS. BINARY PARAMETRIC ERROR ANALYSIS")
print("="*70)

# --- USER CONFIGURABLE STRATEGY & GRID MATRICES ---
STRATEGY = "prep_masked"  # Feel free to change to 'prep_standard', 'prep_full', etc.
VECTORIZERS = ["BoW", "TF-IDF"]
NGRAM_RANGES = [(1, 1), (1, 2), (1, 3), (1, 4)]
SAMPLERS = {
    "SMOTE": SMOTE(random_state=42),
    "Tomek": TomekLinks(),
    "None": None
}

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone

from config import StrategyName, VectorizerName, SamplerName, ModelName, TargetCol, enforce_reproducibility

class ComparatorPipeline:
    def __init__(self, df, strategy: StrategyName):
        self.df = df
        self.strategy = strategy
        self.all_errors = []
        self.performance_metrics = []
        enforce_reproducibility()

    def _get_vectorizer(self, vec_type: VectorizerName, ngram):
        pattern = r'(?u)\[?\b\w[-\\w\.]*\\b\]?'
        if vec_type == VectorizerName.BOW:
            return CountVectorizer(ngram_range=ngram, token_pattern=pattern, max_features=5000)
        return TfidfVectorizer(ngram_range=ngram, token_pattern=pattern, max_features=5000)

    def run_evaluation(self, tasks, base_models, ngrams, samplers):
        for task_name, ctx in tasks.items():
            target_df, encoder = ctx["data"], ctx["encoder"]

            for m_name, base_model in base_models.items():
                for vec_type in [VectorizerName.BOW, VectorizerName.TFIDF]:
                    for ngram in ngrams:
                        for s_name, s_inst in samplers.items():
                            self._evaluate_single_config(task_name, target_df, encoder, m_name, base_model, vec_type, ngram, s_name, s_inst)

    def _evaluate_single_config(self, task_name, df, encoder, m_name, base_model, vec_type, ngram, s_name, s_inst):
        X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
            df[self.strategy], df[TargetCol.SENTIMENT], df.index, test_size=0.2, random_state=42, stratify=df[TargetCol.SENTIMENT]
        )

        vect = self._get_vectorizer(vec_type, ngram)
        X_train_vec = vect.fit_transform(X_train)
        X_test_vec = vect.transform(X_test)

        if s_inst:
            X_train_res, y_train_res = clone(s_inst).fit_resample(X_train_vec, y_train)
        else:
            X_train_res, y_train_res = X_train_vec, y_train

        clf = clone(base_model)
        clf.fit(X_train_res, y_train_res)
        y_pred = clf.predict(X_test_vec)

        self.performance_metrics.append({
            "Task Setting": task_name, "Model Group": m_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Macro-F1": f1_score(y_test, y_pred, average='macro', zero_division=0)
        })

        # Error tracking
        mask = y_test != y_pred
        if np.any(mask):
            self.all_errors.append(pd.DataFrame({
                'Task_Setting': task_name, 'Pipeline_Configuration': f"{m_name}_{vec_type}_{ngram}_{s_name}",
                'Original_Sentence': df.loc[idx_test[mask], 'sentence'].values,
                'True_Label': encoder.inverse_transform(y_test[mask]),
                'Predicted_Label': encoder.inverse_transform(y_pred[mask])
            }))


pipeline_context = SentimentExperimentPipeline(data_path="Sentences_50Agree.txt", target_col="sentiment")
text_preprocessor = Preprocessor(download_resources=True)

# Generate underlying dataframes and configure encoders safely
df = pipeline_context.load_and_initialize_data(preprocessor_instance=text_preprocessor)
label_encoder = pipeline_context.label_encoder
TARGET_COL = "sentiment"

# Define foundational model configurations natively
base_models = {
    "Naive Bayes": MultinomialNB(),
    "FFNN": PyTorchMLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=0.0001,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=50,
        random_state=42
    )
}

custom_token_pattern = r'(?u)\[?\b\w[-\w\.]*\b\]?'
all_errors = []
performance_metrics = []

if STRATEGY in df.columns:
    # -------------------------------------------------------------
    # SETUP 1: MULTICLASS DATASET (Original)
    # -------------------------------------------------------------
    df_multi = df.dropna(subset=[STRATEGY, TARGET_COL]).reset_index(drop=False)

    # -------------------------------------------------------------
    # SETUP 2: BINARY DATASET (Remove Neutral & Re-index Classes)
    # -------------------------------------------------------------
    neutral_idx = list(label_encoder.classes_).index('neutral')
    df_binary_raw = df_multi[df_multi[TARGET_COL] != neutral_idx].copy().reset_index(drop=False)

    # Re-map targets back to continuous [0, 1] for binary execution stability
    binary_label_encoder = LabelEncoder()
    df_binary_raw[TARGET_COL] = binary_label_encoder.fit_transform(binary_label_encoder.inverse_transform(df_binary_raw[TARGET_COL]))

    tasks = {
        "Multiclass": {"data": df_multi, "encoder": label_encoder},
        "Binary (No Neutral)": {"data": df_binary_raw, "encoder": binary_label_encoder}
    }

    print(f"Base Checks -> Multiclass Samples: {len(df_multi)} | Binary Samples: {len(df_binary_raw)}")
    print("-" * 70)

    for task_name, task_context in tasks.items():
        print(f"\n>>> Running Pipeline Evaluations for Task Domain: {task_name.upper()} <<<")
        target_df = task_context["data"]
        current_encoder = task_context["encoder"]

        # --- GRID SEARCH LOOP ---
        for model_name, base_model in base_models.items():
            for vec_type in VECTORIZERS:
                for ngram in NGRAM_RANGES:
                    for sampler_name, sampler_instance in SAMPLERS.items():

                        # Set precise unique pipeline tracking key
                        pipeline_key = f"{model_name}_{vec_type}_NGram_{ngram}_{sampler_name}"
                        model_label = f"{model_name} ({vec_type} | {ngram} | {sampler_name})"

                        X_train_raw, X_test_raw, y_train, y_test, _, idx_test = train_test_split(
                            target_df[STRATEGY], target_df[TARGET_COL], target_df.index,
                            test_size=0.20, random_state=42, stratify=target_df[TARGET_COL]
                        )

                        # Feature Extraction Setup
                        if vec_type == "BoW":
                            vect = CountVectorizer(ngram_range=ngram, token_pattern=custom_token_pattern, max_features=5000)
                        else:
                            vect = TfidfVectorizer(ngram_range=ngram, token_pattern=custom_token_pattern, max_features=5000)

                        X_train_vec = vect.fit_transform(X_train_raw)
                        X_test_vec = vect.transform(X_test_raw)

                        # Apply Mapped Sampler
                        if sampler_instance is not None:
                            # Use clone to guarantee zero state cross-contamination across tasks
                            current_sampler = clone(sampler_instance)
                            X_train_res, y_train_res = current_sampler.fit_resample(X_train_vec, y_train)
                        else:
                            X_train_res, y_train_res = X_train_vec, y_train

                        # Safe Model Execution Guard
                        if model_name == "FFNN":
                            if 'set_reproducible_seeds' in globals():
                                set_reproducible_seeds(42)

                        clf = clone(base_model)
                        clf.fit(X_train_res, y_train_res)
                        y_pred = clf.predict(X_test_vec)

                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

                        performance_metrics.append({
                            "Task Setting": task_name,
                            "Model Identity": pipeline_key,
                            "Model Group": model_name,
                            "Accuracy": acc,
                            "Macro-F1": f1
                        })

                        # Error Extraction Step
                        errors_mask = y_test != y_pred
                        if np.any(errors_mask):
                            error_indices = idx_test[errors_mask]
                            true_labels_text = current_encoder.inverse_transform(target_df.loc[error_indices, TARGET_COL])
                            pred_labels_text = current_encoder.inverse_transform(y_pred[errors_mask])

                            errors_df = pd.DataFrame({
                                'Task_Setting': task_name,
                                'Pipeline_Configuration': model_label,
                                'Original_Sentence': target_df.loc[error_indices, 'sentence'].values,
                                'Preprocessed_Text': target_df.loc[error_indices, STRATEGY].values,
                                'True_Label': true_labels_text,
                                'Predicted_Label': pred_labels_text
                            })
                            all_errors.append(errors_df)

    # -------------------------------------------------------------
    # GENERATE DYNAMIC HIGH-DENSITY GRAPHIC
    # -------------------------------------------------------------
    metrics_df = pd.DataFrame(performance_metrics)

    # Filter for top architectures per model group to prevent messy over-crowded charts
    top_performers = metrics_df.loc[metrics_df.groupby(["Task Setting", "Model Group"])["Macro-F1"].idxmax()]
    melted_metrics = top_performers.melt(id_vars=["Task Setting", "Model Group"], value_vars=["Accuracy", "Macro-F1"], var_name="Metric", value_name="Score")

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=melted_metrics,
        kind="bar",
        x="Model Group",
        y="Score",
        hue="Task Setting",
        col="Metric",
        palette={"Multiclass": "#4A90E2", "Binary (No Neutral)": "#50E3C2"},
        edgecolor="black",
        alpha=0.9,
        height=5,
        aspect=1.2
    )

    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Performance Score Range")

    g.fig.suptitle(f"Task Complexity Impact: Multiclass vs Binary ({STRATEGY.upper()})", weight="bold", fontsize=14, y=1.05)
    plt.savefig(f"multiclass_vs_binary_{STRATEGY}_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    # -------------------------------------------------------------
    # EXPORT REPORTING
    # -------------------------------------------------------------
    if all_errors:
        master_errors_df = pd.concat(all_errors, ignore_index=True)
        master_errors_df.to_csv("Misclassified_Samples_Log.csv", index=False, sep=";")
        print("\n" + "="*70)
        print(f"[+] Success: Comprehensive error logs saved down to 'Misclassified_Samples_Log.csv'")
        print("="*70)

        print("\n" + "#"*45 + "\n--- TARGETED SYSTEMATIC SAMPLE ERRORS ---\n" + "#"*45)
        for setting in ["Multiclass", "Binary (No Neutral)"]:
            print(f"\n" + "="*15 + f" {setting.upper()} SAMPLE DEVIATION ERROR " + "="*15)
            setting_subset = master_errors_df[master_errors_df['Task_Setting'] == setting]

            sampled_pipelines = setting_subset['Pipeline_Configuration'].unique()
            # Sample up to 2 unique broken configs to display as terminal printouts safely
            for model_lbl in sampled_pipelines[:2]:
                final_subset = setting_subset[setting_subset['Pipeline_Configuration'] == model_lbl]

                if not final_subset.empty:
                    row = final_subset.sample(1, random_state=42).iloc[0]
                    print(f"\nPipeline Matrix Ref: {row['Pipeline_Configuration']}")
                    print(f"True Target Ground-Truth: [{row['True_Label'].upper()}]")
                    print(f"Model Predicted Target:  [{row['Predicted_Label'].upper()}]")
                    print(f"Raw Input Text Segment:  {row['Original_Sentence']}")
                    print(f"Engine Cleaned Output:   {row['Preprocessed_Text']}")
                    print("-" * 65)
    else:
        print("\n[!] Outstanding performance! Zero classification errors caught across test splits.")
else:
    print(f"Configuration Error: Preprocessing column '{STRATEGY}' missing from root dataframe.")
