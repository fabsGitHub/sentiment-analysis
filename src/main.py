import os
import ast
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

from config import enforce_reproducibility, GLOBAL_SEED, StrategyName, VectorizerName, SamplerName, ModelName, TargetCol
from Preprocessor import Preprocessor
from SentimentAnalyizer import SentimentExperimentPipeline
from RandomGridSearch import RandomGridSearch
from DataGenerator import DataGenerator
from FFNN import PyTorchMLPClassifier
from MulticlassBinaryComparator import MulticlassBinaryComparator

DEBUG_MODE = True
CACHE_FILEPATH = "data/Master_Grid_Search_Logs.csv"

def main():
    print("===================================================================")
    print("   NLP PROJECT 1.1: PROFOUND EXECUTION CONTROLLER")
    print("===================================================================")

    # lock everything
    enforce_reproducibility(GLOBAL_SEED)
    os.makedirs("data", exist_ok=True)

    # do prep
    print("\n[-->] Executing Task 2: Dataset Ingestion & Preprocessing...")
    base_pipeline = SentimentExperimentPipeline(data_path="data/Sentences_50Agree.txt", target_col=TargetCol.SENTIMENT)
    text_preprocessor = Preprocessor(download_resources=True)

    df_mapped = base_pipeline.load_and_initialize_data(preprocessor_instance=text_preprocessor)
    print(f"[+] Preprocessing complete. Dataset shape: {df_mapped.shape}")

    # parameters grid config
    search_space = {
        ModelName.NAIVE_BAYES: {
            "class": MultinomialNB,
            "param_grid": {
                "alpha": [0.01, 0.1, 0.5, 1.0],
                "fit_prior": [True, False]
            }
        },
        ModelName.FFNN: {
            "class": PyTorchMLPClassifier,
            "param_grid": {
                "hidden_layer_sizes": [(128, 64), (64, 32), (64, 64), (64,)],
                "activation": ["relu"],
                "alpha": [0.0001, 0.001],
                "batch_size": [64, 128],
                "learning_rate_init": [0.001, 0.005],
                "max_iter": [100],
                "patience": [10],
                "random_state": [GLOBAL_SEED],
                "early_stopping": [True],
            }
        }
    }

    strategies = [StrategyName.STANDARD, StrategyName.FULL, StrategyName.MASKED, StrategyName.STANDARD_NUMBERS]
    vectorizers = [VectorizerName.BOW, VectorizerName.TFIDF]
    ngrams = [(1, 1), (1, 2), (1, 3), (1, 4)]
    samplers = {
        SamplerName.NONE: None,
        SamplerName.SMOTE: SMOTE(random_state=GLOBAL_SEED),
        SamplerName.SMOTE_TOMEK: SMOTETomek(random_state=GLOBAL_SEED),
        SamplerName.TOMEK: TomekLinks()
    }

    results_df = None
    if DEBUG_MODE and os.path.exists(CACHE_FILEPATH):
        print(f"\n[-->] DEBUG INTERMEDIATE CACHE HOOK: Loading cached results from '{CACHE_FILEPATH}'...")
        results_df = pd.read_csv(CACHE_FILEPATH, sep=";")
        print(f"[+] Cache loaded successfully. Matrix row metrics footprint: {len(results_df)} entries.")
    else:
        print("\n[-->] Executing Task 3: Multiclass Sentiment Classification Grid Search...")
        search_engine = RandomGridSearch(df=df_mapped, target_col=TargetCol.SENTIMENT, random_state=GLOBAL_SEED)

        results_df = search_engine.fit(
            models_grid=search_space,
            preprocessors=strategies,
            vectorizers=vectorizers,
            ngram_ranges=ngrams,
            samplers=samplers,
            n_iter_per_hyperparam=10
        )
        results_df.to_csv(CACHE_FILEPATH, index=False, sep=";")
        print("[+] Grid search complete. New intermediate cache logs saved to disk.")

    # asset generator stuff
    print("\n[-->] Executing Report Asset Generation...")
    generator = DataGenerator(results_df=results_df, raw_df=df_mapped, target_col=TargetCol.SENTIMENT)

    generator.generate_stacked_latex_table(output_path="data/task2_preprocessing_examples.tex")
    generator.generate_publication_plots(csv_output="data/task3_evaluation_matrix.csv", tex_output="data/task3_figures.tex")

    generator.generate_top_performers_confusion_matrices(
        label_encoder_classes=list(base_pipeline.label_encoder.classes_),
        py_torch_mlp_class=PyTorchMLPClassifier
    )

    # find winning profiles
    print("\n[-->] Dynamically mining the optimal configuration profiles for Error Analysis...")
    target_configs = []

    model_class_map = {
        ModelName.NAIVE_BAYES: MultinomialNB,
        ModelName.FFNN: PyTorchMLPClassifier
    }

    for model_enum, model_class in model_class_map.items():
        model_mask = results_df["Model"] == model_enum.value
        model_subset = results_df[model_mask]

        if not model_subset.empty:
            best_row = model_subset.loc[model_subset["Macro-F1"].idxmax()]
            parsed_params = ast.literal_eval(best_row["Sampled_Hyperparameters"])
            parsed_ngram = ast.literal_eval(best_row["N-Gram"])

            best_model_instance = model_class(**parsed_params)

            target_configs.append({
                "Model Name": model_enum.value,
                "Model": best_model_instance,
                "Vectorizer": best_row["Vectorizer"],
                "N-Gram": parsed_ngram
            })
            print(f"    [+] Chosen {model_enum.value} Profile (Recreated): Macro-F1={best_row['Macro-F1']:.4f}")

    if len(target_configs) == 2:
        generator.generate_disagreement_analysis(
            label_encoder=base_pipeline.label_encoder,
            target_configs=target_configs,
            strategy=StrategyName.MASKED,
            output_path="data/task3_error_analysis.tex"
        )

    # run target space comparison
    print("\n[-->] Executing Comparative Evaluation: Multiclass vs. Binary Target Space Task Structures...")

    comparator = MulticlassBinaryComparator(
        df=df_mapped,
        label_encoder=base_pipeline.label_encoder,
        strategy=StrategyName.MASKED,
        random_state=GLOBAL_SEED
    )

    comparator.run_comparative_matrix(
        models_grid=search_space,
        vectorizers=vectorizers,
        ngram_ranges=ngrams,
        samplers=samplers,
        task3_results_df=results_df
    )

    print("\n[-->] Generating Optimal Hyperparameter LaTeX Tables...")
    generator.export_optimal_hyperparameter_tables_3_tables(comparator=comparator)

    print("\n[-->] Generating Multiclass vs. Binary Structural Comparison Matrix Table...")
    generator.generate_multiclass_binary_comparison_table(comparator=comparator, output_path="data/multiclass_binary_comparison.tex")

    print("\n===================================================================")
    print("    NLP EXPERIMENT WORKFLOW COMPLETED SUCCESSFULLY")
    print("===================================================================")

if __name__ == "__main__":
    main()
