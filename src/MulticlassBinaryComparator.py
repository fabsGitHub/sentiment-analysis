import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone

from config import StrategyName, VectorizerName, TargetCol, enforce_reproducibility

class MulticlassBinaryComparator:
    """
    Orchestrates parametric comparison evaluation structures across cross-combinatorial
    Multiclass vs. Binary experimental tasks, generating reporting logs and metrics plots.
    """
    def __init__(self, df: pd.DataFrame, label_encoder: LabelEncoder, strategy: StrategyName = StrategyName.MASKED, random_state: int = 42):
        self.df = df.copy()
        self.label_encoder = label_encoder
        self.strategy = strategy.value if hasattr(strategy, 'value') else str(strategy)
        self.random_state = random_state
        self.all_errors = []
        self.performance_metrics = []
        enforce_reproducibility(self.random_state)

    def run_comparative_matrix(self, models_grid: dict, vectorizers: list, ngram_ranges: list, samplers: dict):
        if self.strategy not in self.df.columns:
            print(f" [!] Configuration Error: Preprocessing column '{self.strategy}' missing from root dataframe.")
            return

        target_col = TargetCol.SENTIMENT.value if hasattr(TargetCol.SENTIMENT, 'value') else str(TargetCol.SENTIMENT)
        custom_token_pattern = r'(?u)\[?\b\w[-\w\.]*\b\]?'

        # -------------------------------------------------------------
        # SETUP 1: MULTICLASS TASK CONFIGURATION
        # -------------------------------------------------------------
        df_multi = self.df.dropna(subset=[self.strategy, target_col]).reset_index(drop=True)

        # -------------------------------------------------------------
        # SETUP 2: BINARY TASK CONFIGURATION (Omit Neutrals)
        # -------------------------------------------------------------
        try:
            neutral_idx = list(self.label_encoder.classes_).index('neutral')
        except ValueError:
            neutral_idx = 1  # Fallback guess index if not cleanly fitted

        df_binary_raw = df_multi[df_multi[target_col] != neutral_idx].copy().reset_index(drop=True)
        binary_label_encoder = LabelEncoder()
        df_binary_raw[target_col] = binary_label_encoder.fit_transform(
            binary_label_encoder.inverse_transform(df_binary_raw[target_col])
        )

        tasks = {
            "Multiclass": {"data": df_multi, "encoder": self.label_encoder},
            "Binary (No Neutral)": {"data": df_binary_raw, "encoder": binary_label_encoder}
        }

        print(f"\n[-->] Starting Comparative Grid: Multiclass Samples={len(df_multi)} | Binary Samples={len(df_binary_raw)}")
        print("-" * 80)

        for task_name, task_context in tasks.items():
            print(f" [+] Processing Task Complex Setting Domain: {task_name.upper()}")
            target_df = task_context["data"]
            current_encoder = task_context["encoder"]

            # Unpack structural settings from models_grid configuration matrix setup
            for model_enum, model_meta in models_grid.items():
                model_str = model_enum.value if hasattr(model_enum, 'value') else str(model_enum)
                model_class = model_meta["class"]
                param_grid = model_meta["param_grid"]

                # Resolve standard configuration parameters (takes first available option)
                base_params = {}
                for param_name, param_values in param_grid.items():
                    if isinstance(param_values, list) and len(param_values) > 0:
                        base_params[param_name] = param_values[0]
                    else:
                        base_params[param_name] = param_values

                base_model = model_class(**base_params)

                for vec_enum in vectorizers:
                    vec_str = vec_enum.value if hasattr(vec_enum, 'value') else str(vec_enum)

                    for ngram in ngram_ranges:
                        for sampler_key, sampler_instance in samplers.items():
                            sampler_str = sampler_key.value if hasattr(sampler_key, 'value') else str(sampler_key)

                            pipeline_key = f"{model_str}_{vec_str}_NGram_{ngram}_{sampler_str}"
                            model_label = f"{model_str} ({vec_str} | {ngram} | {sampler_str})"

                            # Split training matrices securely per setting iteration
                            X_train_raw, X_test_raw, y_train, y_test, _, idx_test = train_test_split(
                                target_df[self.strategy], target_df[target_col], target_df.index,
                                test_size=0.20, random_state=self.random_state, stratify=target_df[target_col]
                            )

                            if vec_str == "BoW":
                                vect = CountVectorizer(ngram_range=ngram, token_pattern=custom_token_pattern, max_features=5000)
                            else:
                                vect = TfidfVectorizer(ngram_range=ngram, token_pattern=custom_token_pattern, max_features=5000)

                            try:
                                X_train_vec = vect.fit_transform(X_train_raw)
                                X_test_vec = vect.transform(X_test_raw)
                            except Exception:
                                continue

                            if sampler_instance is not None:
                                try:
                                    X_train_res, y_train_res = clone(sampler_instance).fit_resample(X_train_vec, y_train)
                                except Exception:
                                    X_train_res, y_train_res = X_train_vec, y_train
                            else:
                                X_train_res, y_train_res = X_train_vec, y_train

                            try:
                                clf = clone(base_model)
                                clf.fit(X_train_res, y_train_res)
                                y_pred = clf.predict(X_test_vec)

                                acc = accuracy_score(y_test, y_pred)
                                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

                                self.performance_metrics.append({
                                    "Task Setting": task_name,
                                    "Model Identity": pipeline_key,
                                    "Model Group": model_str,
                                    "Accuracy": acc,
                                    "Macro-F1": f1
                                })

                                # Tracking specific misclassifications
                                errors_mask = y_test != y_pred
                                if np.any(errors_mask):
                                    error_indices = idx_test[errors_mask]
                                    true_labels_text = current_encoder.inverse_transform(target_df.loc[error_indices, target_col])
                                    pred_labels_text = current_encoder.inverse_transform(y_pred[errors_mask])

                                    errors_df = pd.DataFrame({
                                        'Task_Setting': task_name,
                                        'Pipeline_Configuration': model_label,
                                        'Original_Sentence': target_df.loc[error_indices, 'sentence'].values,
                                        'Preprocessed_Text': target_df.loc[error_indices, self.strategy].values,
                                        'True_Label': true_labels_text,
                                        'Predicted_Label': pred_labels_text
                                    })
                                    self.all_errors.append(errors_df)
                            except Exception as e:
                                print(f" [!] Error fitting configuration structural step: {str(e)}")
                                continue

        # -------------------------------------------------------------
        # METRIC REPORT PROCESSING AND GRAPHICS DISK EXPORT
        # -------------------------------------------------------------
        if not self.performance_metrics:
            print(" [!] Error Matrix Generation aborted: No valid execution metrics recorded.")
            return

        metrics_df = pd.DataFrame(self.performance_metrics)
        top_performers = metrics_df.loc[metrics_df.groupby(["Task Setting", "Model Group"])["Macro-F1"].idxmax()]
        melted_metrics = top_performers.melt(id_vars=["Task Setting", "Model Group"], value_vars=["Accuracy", "Macro-F1"], var_name="Metric", value_name="Score")

        sns.set_theme(style="whitegrid")
        g = sns.catplot(
            data=melted_metrics, kind="bar", x="Model Group", y="Score", hue="Task Setting", col="Metric",
            palette={"Multiclass": "#4A90E2", "Binary (No Neutral)": "#50E3C2"}, edgecolor="black", alpha=0.9, height=5, aspect=1.2
        )

        for ax in g.axes.flat:
            for container in ax.containers:
                ax.bar_label(container, fmt="%.3f", padding=3, fontsize=10, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Performance Score Range")

        g.fig.suptitle(f"Task Complexity Impact: Multiclass vs Binary ({self.strategy.upper()})", weight="bold", fontsize=14, y=1.05)
        plt.savefig(f"../data/multiclass_vs_binary_{self.strategy}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # -------------------------------------------------------------
        # EXPORT CONSOLIDATED ERROR LOGS DATASETS
        # -------------------------------------------------------------
        if self.all_errors:
            master_errors_df = pd.concat(self.all_errors, ignore_index=True)
            master_errors_df.to_csv("../data/Misclassified_Samples_Log.csv", index=False, sep=";")
            print(f" [+] Success: Comparative error logs exported to 'data/Misclassified_Samples_Log.csv'")

            print("\n" + "#"*50 + "\n--- TARGETED SYSTEMATIC SAMPLE ERRORS ---\n" + "#"*50)
            for setting in ["Multiclass", "Binary (No Neutral)"]:
                print(f"\n" + "="*15 + f" {setting.upper()} SAMPLE DEVIATION ERROR " + "="*15)
                setting_subset = master_errors_df[master_errors_df['Task_Setting'] == setting]
                sampled_pipelines = setting_subset['Pipeline_Configuration'].unique()

                for model_lbl in sampled_pipelines[:2]:
                    final_subset = setting_subset[setting_subset['Pipeline_Configuration'] == model_lbl]
                    if not final_subset.empty:
                        row = final_subset.sample(1, random_state=self.random_state).iloc[0]
                        print(f"\nPipeline Matrix Ref: {row['Pipeline_Configuration']}")
                        print(f"True Target Ground-Truth: [{row['True_Label'].upper()}]")
                        print(f"Model Predicted Target:  [{row['Predicted_Label'].upper()}]")
                        print(f"Raw Input Text Segment:  {row['Original_Sentence']}")
                        print(f"Engine Cleaned Output:   {row['Preprocessed_Text']}")
                        print("-" * 65)
        else:
            print("\n [!] Outstanding system state performance: Zero error instances isolated.")
