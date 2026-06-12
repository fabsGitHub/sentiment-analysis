import os
import re
import ast
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import clone

from config import *

class DataGenerator:
    def __init__(self, results_df: pd.DataFrame, raw_df: pd.DataFrame, target_col: str = TargetCol.SENTIMENT):
        self.results_df = results_df.copy()
        self.df = raw_df.copy()
        self.target_col = target_col

        self.sampler_map = {
            SamplerName.NONE: None,
            SamplerName.SMOTE: SMOTE(random_state=GLOBAL_SEED),
            SamplerName.SMOTE_TOMEK: SMOTETomek(random_state=GLOBAL_SEED),
            SamplerName.TOMEK: TomekLinks()
        }

        self.strategy_colors = {
            StrategyName.STANDARD: '#4A90E2',
            StrategyName.STANDARD_NUMBERS: '#9B5DE5',
            StrategyName.FULL: '#50E3C2',
            StrategyName.MASKED: '#E2844A',
        }
        self.strategy_order = [StrategyName.STANDARD, StrategyName.FULL, StrategyName.MASKED, StrategyName.STANDARD_NUMBERS]
        self.custom_token_pattern = r'(?u)\[?\b\w[-\w\.]*\b\]?'

    @staticmethod
    def generate_latex_hyperparameter_table(caption: str, label: str, params: dict, is_nb: bool = False) -> str:
        # Unified mapping dictionary covering both model configuration frameworks
        param_labels = {
            # Naive Bayes Parameters
            "alpha": r"Smoothing Parameter ($\alpha$)" if is_nb else r"L2 Penalty ($\alpha$)",
            "fit_prior": "Learn Class Prior Probabilities",
            # FFNN Core Engine
            "hidden_layer_sizes": "Hidden Layer Sizes",
            "activation": "Activation Function",
            "solver": "Optimization Solver",
            # FFNN Training Dynamics
            "batch_size": "Batch Size",
            "learning_rate_init": "Initial Learning Rate",
            "max_iter": "Maximum Iterations",
            "early_stopping": "Early Stopping",
            "validation_fraction": "Validation Fraction",
            "random_state": "Random State Seed"
        }

        latex = r"""\begin{table}[H]
    \centering
    \caption{""" + caption + r"""}
    \label{""" + label + r"""}
    \small
    \begin{tabular}{ll}
    \toprule
    \textbf{Hyperparameter} & \textbf{Value} \\
    \midrule
"""
        if is_nb:
            # Naive Bayes Structure Layout
            nb_keys = ["alpha", "fit_prior"]
            for key in nb_keys:
                if key in params:
                    latex += f"    {param_labels[key]} & {params[key]} \\\\\n"
        else:
            # FFNN Structural Architecture Layout
            core_engine_keys = ["hidden_layer_sizes", "activation", "solver", "alpha"]
            training_dynamics_keys = ["batch_size", "learning_rate_init", "max_iter", "early_stopping", "validation_fraction", "random_state"]

            latex += r"    \multicolumn{2}{l}{\textit{Architecture \& Core Engine}} \\" + "\n"
            for key in core_engine_keys:
                if key in params:
                    latex += f"    {param_labels[key]} & {params[key]} \\\\\n"

            latex += r"    \midrule" + "\n"
            latex += r"    \multicolumn{2}{l}{\textit{Training Dynamics}} \\" + "\n"
            for key in training_dynamics_keys:
                if key in params:
                    latex += f"    {param_labels[key]} & {params[key]} \\\\\n"

        latex += r"""    \bottomrule
    \end{tabular}
    \end{table}
"""
        return latex

    def export_optimal_hyperparameter_tables_3_tables(self, comparator):
        """
        Extracts and converts optimal training parameters into formal publication LaTeX tables
        for both FFNN and Naive Bayes architectures across multi-task execution contexts.
        """

        models_to_process = [
            {"enum_val": ModelName.FFNN.value, "file_suffix": "ffnn", "is_nb": False},
            {"enum_val": ModelName.NAIVE_BAYES.value, "file_suffix": "mnb", "is_nb": True}
        ]

        for target in models_to_process:
            model_results = self.results_df[self.results_df["Model"] == target["enum_val"]]
            if not model_results.empty:
                best_row = model_results.loc[model_results["Macro-F1"].idxmax()]
                best_params = ast.literal_eval(best_row["Sampled_Hyperparameters"])

                # Apply environmental fallbacks only if processing the Neural Network configuration
                if not target["is_nb"]:
                    best_params.setdefault("solver", "adam")
                    best_params.setdefault("early_stopping", True)
                    best_params.setdefault("validation_fraction", 0.1)

                caption = f"Optimal Hyperparameter Configuration for the {target['enum_val']} Model via Task 3 Grid Search (Strategy: {best_row['Strategy']}, Vectorizer: {best_row['Vectorizer']})"
                latex_table = self.generate_latex_hyperparameter_table(
                    caption=caption,
                    label=f"tab:task3_{target['file_suffix']}_hyperparameters",
                    params=best_params,
                    is_nb=target["is_nb"]
                )

                output_file = f"data/task3_best_hyperparameters_{target['file_suffix']}.tex"
                with open(output_file, "w") as f:
                    f.write(latex_table)
                print(f"    [+] Exported Multiclass Table: {output_file}")

        if hasattr(comparator, 'performance_metrics') and comparator.performance_metrics:
            comp_metrics_df = pd.DataFrame(comparator.performance_metrics)

            for target in models_to_process:
                comp_subset = comp_metrics_df[comp_metrics_df["Model"] == target["enum_val"]]
                if not comp_subset.empty:
                    best_comp_row = comp_subset.loc[comp_subset["Macro-F1"].idxmax()]

                    # Resilient extraction of stored parameter strings
                    if "Sampled_Hyperparameters" in best_comp_row:
                        raw_params = best_comp_row["Sampled_Hyperparameters"]
                    elif "Hyperparameters" in best_comp_row:
                        raw_params = best_comp_row["Hyperparameters"]
                    else:
                        raw_params = {}

                    final_comp_params = ast.literal_eval(raw_params) if isinstance(raw_params, str) else raw_params

                    if not target["is_nb"]:
                        final_comp_params.setdefault("solver", "adam")
                        final_comp_params.setdefault("early_stopping", True)
                        final_comp_params.setdefault("validation_fraction", 0.1)

                    comp_caption = f"Optimal Hyperparameter Configuration for the {target['enum_val']} Model via Comparative Matrix Analysis ({best_comp_row['Task Setting']} Domain)"
                    comp_latex = self.generate_latex_hyperparameter_table(
                        caption=comp_caption,
                        label=f"tab:comparator_{target['file_suffix']}_hyperparameters",
                        params=final_comp_params,
                        is_nb=target["is_nb"]
                    )

                    output_file = f"data/comparator_best_hyperparameters_{target['file_suffix']}.tex"
                    with open(output_file, "w") as f:
                        f.write(comp_latex)
                    print(f"    [+] Exported Comparator Table: {output_file}")
        else:
            print("    [!] Warning: No data found inside comparator tracking loops to extract configuration fields.")
    def generate_top_performers_confusion_matrices(self, label_encoder_classes: list, py_torch_mlp_class = None):
        print("\n" + "="*50)
        print("GENERATING CONFUSION MATRICES FOR TOP PERFORMERS")
        print("="*50)

        top_models = []
        for model_enum in [ModelName.NAIVE_BAYES, ModelName.FFNN]:
            m_val = model_enum.value
            model_subset = self.results_df[self.results_df['Model'] == m_val]
            if not model_subset.empty:
                best_row = model_subset.loc[model_subset['Macro-F1'].idxmax()]
                top_models.append(best_row)

        for config in top_models:
            model_name = config['Model']
            strategy = config['Strategy']
            ngram = ast.literal_eval(config['N-Gram'])
            sampled_params = ast.literal_eval(config['Sampled_Hyperparameters'])

            sampler_name = config['Sampling Strategy']
            sampler = None
            for k, v in self.sampler_map.items():
                if k.value == sampler_name:
                    sampler = v
                    break

            print(f"\nRe-evaluating reproducibility for: {model_name}...")

            df_clean = self.df.dropna(subset=[strategy, self.target_col]).reset_index(drop=True)
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                df_clean[strategy], df_clean[self.target_col],
                test_size=0.20, random_state=GLOBAL_SEED, stratify=df_clean[self.target_col]
            )

            if config['Vectorizer'] == VectorizerName.BOW.value:
                vect = CountVectorizer(ngram_range=ngram, token_pattern=self.custom_token_pattern, max_features=5000)
            else:
                vect = TfidfVectorizer(ngram_range=ngram, token_pattern=self.custom_token_pattern, max_features=5000)

            X_train_vec = vect.fit_transform(X_train_raw)
            X_test_vec = vect.transform(X_test_raw)

            if sampler is not None:
                X_train_res, y_train_res = clone(sampler).fit_resample(X_train_vec, y_train)
            else:
                X_train_res, y_train_res = X_train_vec, y_train

            if model_name == ModelName.NAIVE_BAYES.value:
                clf = MultinomialNB(**sampled_params)
            else:
                if py_torch_mlp_class is None:
                    print("Skipping FFNN Matrix: Missing valid PyTorchMLPClassifier reference.")
                    continue
                if "random_state" not in sampled_params:
                    sampled_params["random_state"] = GLOBAL_SEED
                clf = py_torch_mlp_class(**sampled_params)

            clf.fit(X_train_res, y_train_res)
            y_pred = clf.predict(X_test_vec)
            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(6.5, 5.5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=label_encoder_classes, yticklabels=label_encoder_classes,
                linewidths=1.5, linecolor='white',
                annot_kws={"size": 15, "weight": "bold"}
            )

            plt.xlabel("Predicted Label", labelpad=10, fontsize=11, weight='bold')
            plt.ylabel("Actual Label", labelpad=10, fontsize=11, weight='bold')

            clean_filename = f"data/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            plt.tight_layout()
            plt.savefig(clean_filename, dpi=300, bbox_inches='tight')
            print(f" -> Matrix generated and saved as: '{clean_filename}'")
            plt.close()

    def generate_stacked_latex_table(self, output_path: str = "data/preprocessing_stacked.tex") -> str:
        masks = [r'\[PHONE\]', r'\[MONEY\]', r'\[PERCENT\]', r'\[DATE\]', r'\[TIME\]', r'\[MEASUREMENT\]', r'\[NUMBER\]']
        mask_pattern = '|'.join(masks)

        dense_mask_regex = re.compile(rf"(?:{mask_pattern}).*?(?:{mask_pattern})", re.IGNORECASE)
        filtered = self.df[
            (self.df['sentence'].str.len() < 100) &
            (self.df[StrategyName.MASKED.value].str.contains(dense_mask_regex, na=False, regex=True))
        ]

        if len(filtered) < 2:
            filtered = self.df.head(2)

        selected_df = filtered.head(2)[['sentence', StrategyName.STANDARD.value, StrategyName.STANDARD_NUMBERS.value, StrategyName.FULL.value, StrategyName.MASKED.value]].copy()
        labels = ['Original Sentence', 'Strategy 1 (Standard)', 'Strategy 2 (Enhanced)', 'Strategy 3 (Full)', 'Strategy 4 (Masked)']

        latex_rows = []
        for idx, row in enumerate(selected_df.itertuples(index=False)):
            latex_rows.append(f"\\textbf{{Example {idx + 1}}} & \\\\")
            for label, val in zip(labels, row):
                escaped_val = str(val).replace("$", "\\$").replace("%", "\\%").replace("_", "\\_")
                latex_rows.append(f"{label} & {escaped_val} \\\\")
            if idx == 0:
                latex_rows.append("\\midrule")

        latex_output = f"""\\begin{{table}}[h]
\\centering
\\caption{{Preprocessing Results Stacked Vertically}}
\\label{{tab:preprocessing_stacked_vertical}}
\\small
\\begin{{tabular}}{{lp{{12cm}}}}
\\toprule
\\textbf{{Feature / Pipeline}} & \\textbf{{Processed Text String Baseline}} \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

        with open(output_path, "w") as f:
            f.write(latex_output)
        return latex_output

    def generate_publication_plots(self, csv_output: str = "data/grand_8d_evaluation_matrix.csv", tex_output: str = "data/figure_titles.tex"):
        self.results_df["Macro-F1"] = pd.to_numeric(self.results_df["Macro-F1"], errors='coerce').fillna(0.0)

        global_winner = self.results_df.loc[self.results_df['Macro-F1'].idxmax()]
        print("\n" + "-"*50 + "\n--- GLOBAL PERFORMANCE WINNER ---\n" + "-"*50)
        print(global_winner[['Model', 'Strategy', 'Sampling Strategy', 'Vectorizer', 'N-Gram', 'Macro-F1']])
        print("-" * 50)

        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 13,
            'xtick.labelsize': 8, 'ytick.labelsize': 10,
        })

        self.results_df["Vec_N-Gram"] = self.results_df["Vectorizer"].astype(str) + "\n" + self.results_df["N-Gram"].astype(str)
        self.results_df["Strategy"] = self.results_df["Strategy"].apply(lambda x: x.value if hasattr(x, 'value') else str(x))

        min_y = self.results_df["Macro-F1"].min()
        max_y = self.results_df["Macro-F1"].max()
        y_lower_bound = max(0.0, float(np.floor(min_y * 20) / 20) - 0.05)
        y_upper_bound = min(1.0, float(np.ceil(max_y * 20) / 20) + 0.05)

        str_strategy_order = [str(strat.value) for strat in self.strategy_order]
        str_strategy_colors = {str(k.value): v for k, v in self.strategy_colors.items()}

        publication_title_map = {
            StrategyName.STANDARD.value: r"Standard Baseline ($\mathcal{S}_1$)",
            StrategyName.STANDARD_NUMBERS.value: r"Enhanced Baseline ($\mathcal{S}_2$)",
            StrategyName.FULL.value: r"Full Preprocessing ($\mathcal{S}_3$)",
            StrategyName.MASKED.value: r"Comprehensive Masking ($\mathcal{S}_4$)"
        }

        self.results_df.to_csv(csv_output, sep=";")
        unique_models = self.results_df["Model"].unique()
        unique_samplings = self.results_df["Sampling Strategy"].unique()

        print(f"\nGenerating publication quality metric comparison plots...")
        latex_captions = []

        for model in unique_models:
            for sampling in unique_samplings:
                subset_data = self.results_df[(self.results_df["Model"] == model) & (self.results_df["Sampling Strategy"] == sampling)].copy()
                if subset_data.empty:
                    continue

                clean_model = model.lower().replace(" ", "_").replace("(", "").replace(")", "")
                clean_sampling = sampling.lower().replace(" ", "_").replace("+", "plus")
                filename = f"data/matrix_{clean_model}_{clean_sampling}.png"

                plt.figure(figsize=(10, 6))
                ax = plt.gca()
                subset_data = subset_data.sort_values(by=["Vec_N-Gram", "Strategy"]).reset_index(drop=True)

                sns.barplot(
                    data=subset_data, x="Vec_N-Gram", y="Macro-F1", hue="Strategy",
                    hue_order=str_strategy_order, palette=str_strategy_colors,
                    edgecolor="black", linewidth=0.6, alpha=0.85, ax=ax
                )

                real_data = subset_data[subset_data['Macro-F1'] > 0.0]
                if not real_data.empty:
                    best_row = real_data.loc[real_data['Macro-F1'].idxmax()]
                    unique_x_labels = sorted(subset_data['Vec_N-Gram'].unique().tolist())
                    if best_row['Vec_N-Gram'] in unique_x_labels:
                        x_pos = unique_x_labels.index(best_row['Vec_N-Gram'])
                        ax.scatter(x_pos, best_row['Macro-F1'] + 0.015, color='red', marker='*', s=120, zorder=5)
                        ax.text(x_pos, best_row['Macro-F1'] + 0.025, f"{best_row['Macro-F1']:.3f}",
                                ha='center', fontsize=9.5, color='red', weight='bold')

                ax.set_xlabel("Configuration Layout (Vectorizer & N-Gram)", labelpad=10)
                ax.set_ylabel("Macro-F1 Score")
                ax.set_ylim(y_lower_bound, y_upper_bound)

                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
                ax.yaxis.grid(True, which='major', linestyle="-", alpha=0.5)
                ax.yaxis.grid(True, which='minor', linestyle="--", alpha=0.25)
                ax.xaxis.grid(True, linestyle=":", alpha=0.4)

                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    formatted_labels = [publication_title_map.get(lbl, lbl) for lbl in labels]
                    ax.legend(handles[:4], formatted_labels[:4], title="Preprocessing Strategy", loc="upper left", frameon=True)

                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()

                latex_captions.append({
                    'file': filename,
                    'title': f"Performance Metrics: {model} ({sampling})",
                    'desc': "Bars = Macro-F1 Performance Metric Analysis"
                })

        with open(tex_output, "w") as f:
            for item in latex_captions:
                f.write(r"\begin{figure}[htbp]" + "\n")
                f.write(r"    \centering" + "\n")
                f.write(f"    \\includegraphics[width=0.8\\textwidth]{{{item['file']}}}" + "\n")
                f.write(f"    \\caption{{{item['title']}: {item['desc']}}}" + "\n")
                f.write(r"    \label{fig:" + item['file'].replace('.png', '').replace('data/', '') + "}" + "\n")
                f.write(r"\end{figure}" + "\n\n")

    def generate_disagreement_analysis(self, label_encoder, target_configs: list, strategy: str = "prep_masked", output_path: str = "data/disagreement_analysis.tex") -> str:
        strategy_str = strategy.value if hasattr(strategy, 'value') else str(strategy)
        df_multi = self.df.dropna(subset=[strategy_str, self.target_col]).reset_index(drop=False)

        all_preds_list = []
        for config in target_configs:
            model_label = config['Model Name']

            X_train_raw, X_test_raw, y_train, y_test, _, idx_test = train_test_split(
                df_multi[strategy_str], df_multi[self.target_col], df_multi.index,
                test_size=0.20, random_state=42, stratify=df_multi[self.target_col]
            )

            if config["Vectorizer"] == VectorizerName.BOW.value:
                vect = CountVectorizer(ngram_range=config["N-Gram"], token_pattern=self.custom_token_pattern)
            else:
                vect = TfidfVectorizer(ngram_range=config["N-Gram"], token_pattern=self.custom_token_pattern)

            X_train_vec = vect.fit_transform(X_train_raw)
            X_test_vec = vect.transform(X_test_raw)

            sampler = SMOTE(random_state=42)
            X_train_res, y_train_res = sampler.fit_resample(X_train_vec, y_train)

            clf = clone(config["Model"])
            clf.fit(X_train_res, y_train_res)
            y_pred = clf.predict(X_test_vec)

            true_labels = label_encoder.inverse_transform(y_test)
            pred_labels = label_encoder.inverse_transform(y_pred)
            raw_sentences = df_multi.loc[idx_test, 'sentence'].values
            masked_sentences = df_multi.loc[idx_test, strategy_str].values

            for src_idx, src_text, mask_text, true_lbl, pred_lbl in zip(idx_test, raw_sentences, masked_sentences, true_labels, pred_labels):
                all_preds_list.append({
                    'Sample_ID': src_idx,
                    'Sentence': src_text,
                    'Masked_Sentence': mask_text,
                    'True Label': true_lbl,
                    'Pipeline': model_label,
                    'Prediction': pred_lbl
                })

        if not all_preds_list:
            return ""

        preds_df = pd.DataFrame(all_preds_list)
        sentence_info = preds_df.drop_duplicates('Sample_ID')[['Sample_ID', 'Sentence', 'True Label', 'Masked_Sentence']]
        p_pred = preds_df.pivot(index='Sample_ID', columns='Pipeline', values='Prediction').reset_index()
        p_pred = p_pred.merge(sentence_info, on='Sample_ID', how='left')

        p_pred = p_pred[p_pred['Sentence'].str.len() < 70].copy()
        selected_rows = []

        cond_1 = (p_pred[ModelName.FFNN.value] == p_pred['True Label']) & (p_pred[ModelName.NAIVE_BAYES.value] != p_pred['True Label'])
        if cond_1.any():
            selected_rows.append(p_pred[cond_1].iloc[0])

        cond_2 = (p_pred[ModelName.FFNN.value] != p_pred['True Label']) & (p_pred[ModelName.NAIVE_BAYES.value] == p_pred['True Label'])
        if cond_2.any():
            selected_rows.append(p_pred[cond_2].iloc[0])

        if len(selected_rows) == 0:
            return ""

        display_df = pd.DataFrame(selected_rows)
        display_df['Sentence'] = display_df.apply(
            lambda r: f"\"{str(r['Sentence']).replace('_', '\\_')}\" \\\\ \\textit{{\"Masked: {str(r['Masked_Sentence']).replace('_', '\\_')}\"}}", axis=1
        )

        latex_code = r"""\begin{table}[h]
\centering
\caption{Model Prediction Disagreement Analysis}\n""" + f"\\begin{{tabular}}{{|p{{8cm}}|p{{2cm}}|p{{2cm}}|p{{2cm}}|}}\n\\hline\n\\textbf{{Sentence / Preprocessed Context}} & \\textbf{{True Label}} & \\textbf{{{ModelName.FFNN.value} Pred.}} & \\textbf{{{ModelName.NAIVE_BAYES.value} Pred.}}  \\\\ \\hline\n"
        for _, row in display_df.iterrows():
            latex_code += f"{row['Sentence']} & {str(row['True Label'])} & {row[ModelName.FFNN.value]} & {row[ModelName.NAIVE_BAYES.value]}  \\\\ \\hline\n"
        latex_code += r"""\end{tabular}
\end{table}"""

        with open(output_path, "w") as f:
            f.write(latex_code)
        return latex_code


    def generate_multiclass_binary_comparison_table(self, comparator, output_path: str = "data/task4_multiclass_binary_comparison.tex") -> str:
        """
        Gathers performance metrics from the comparator to compile a publication-grade
        LaTeX table contrasting Multiclass vs. Binary target spaces across models.
        """
        if not hasattr(comparator, 'performance_metrics') or not comparator.performance_metrics:
            print("    [!] Error: No performance tracking data found inside comparator instance.")
            return ""

        # Convert the raw dictionary lists from the comparator into a clean DataFrame
        comp_df = pd.DataFrame(comparator.performance_metrics)

        # Filter down to look closely at performance across Task Settings and Models
        # Group and find the absolute maximum performance profile achieved for each setting
        idx_max = comp_df.groupby(["Task Setting", "Model"])["Macro-F1"].idxmax()
        summary_df = comp_df.loc[idx_max].sort_values(by=["Model", "Task Setting"], ascending=[True, False])

        latex = r"""\begin{table}[htbp]
    \centering
    \caption{Performance Comparison: Multiclass Sentiment Structure vs. Binary Classification Target Spaces}
    \label{tab:multiclass_binary_comparison}
    \small
    \begin{tabular}{llccc}
    \toprule
    \textbf{Model Framework} & \textbf{Task Domain} & \textbf{Vectorizer} & \textbf{N-Gram} & \textbf{Optimal Macro-F1} \\
    \midrule
"""

        prev_model = None
        for _, row in summary_df.iterrows():
            # If switching to a new model type, add a small visual line separator
            if prev_model is not None and prev_model != row["Model"]:
                latex += "    \\midrule\n"

            # Format the layout metrics safely
            model_disp = row["Model"] if row["Model"] != prev_model else ""
            task_disp = row["Task Setting"]
            vec_disp = row["Vectorizer"]
            ngram_disp = str(row["N-Gram"]).replace("_", "\\_")
            f1_disp = f"{row['Macro-F1']:.4f}"

            latex += f"    {model_disp} & {task_disp} & {vec_disp} & {ngram_disp} & \\textbf{{{f1_disp}}} \\\\\n"
            prev_model = row["Model"]

        latex += r"""    \bottomrule
    \end{tabular}
\end{table}
"""

        # Ensure target data directory exists and commit assets to file structure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex)

        print(f"    [+] Successfully generated comparative summary matrix: {output_path}")
        return latex
