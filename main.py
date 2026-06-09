import os
import ast
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

# Import architectural modules and new configuration Enums
from config import enforce_reproducibility, GLOBAL_SEED, StrategyName, VectorizerName, SamplerName, ModelName, TargetCol
from Preprocessor import Preprocessor
from SentimentAnalyizer import SentimentExperimentPipeline
from RandomGridSearch import RandomGridSearch
from DataGenerator import DataGenerator
from FFNN import PyTorchMLPClassifier

def main():
    print("===================================================================")
    print("   NLP PROJECT 1.1: PROFOUND EXECUTION CONTROLLER")
    print("===================================================================")

    # 1. Lock down all stochastic elements globally
    enforce_reproducibility(GLOBAL_SEED)
    os.makedirs("data", exist_ok=True)

    # -----------------------------------------------------------------
    # TASK 2: TEXT PREPROCESSING
    # -----------------------------------------------------------------
    print("\n[-->] Executing Task 2: Dataset Ingestion & Preprocessing...")
    base_pipeline = SentimentExperimentPipeline(data_path="Sentences_50Agree.txt", target_col=TargetCol.SENTIMENT)
    text_preprocessor = Preprocessor(download_resources=True)

    df_mapped = base_pipeline.load_and_initialize_data(preprocessor_instance=text_preprocessor)
    print(f"[+] Preprocessing complete. Dataset shape: {df_mapped.shape}")

    # -----------------------------------------------------------------
    # TASK 3: SENTIMENT CLASSIFICATION GRID SEARCH
    # -----------------------------------------------------------------
    print("\n[-->] Executing Task 3: Multiclass Sentiment Classification Grid Search...")

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
                "hidden_layer_sizes": [(128, 64), (64, 32), (64, 64)],
                "activation": ["relu"],
                "alpha": [0.0001, 0.001],
                "batch_size": [64, 128],
                "learning_rate_init": [0.001, 0.005],
                "max_iter": [30],
                "random_state": [GLOBAL_SEED] # Explicitly pass the strict seed into the parameter space
            }
        }
    }

    strategies = [StrategyName.STANDARD, StrategyName.FULL, StrategyName.MASKED]
    vectorizers = [VectorizerName.BOW, VectorizerName.TFIDF]
    ngrams = [(1, 1), (1, 2), (1, 3)]
    samplers = {
        SamplerName.NONE: None,
        SamplerName.SMOTE: SMOTE(random_state=GLOBAL_SEED),
        SamplerName.TOMEK: TomekLinks()
    }

    search_engine = RandomGridSearch(df=df_mapped, target_col=TargetCol.SENTIMENT, random_state=GLOBAL_SEED)
    results_df = search_engine.fit(
        models_grid=search_space,
        preprocessors=strategies,
        vectorizers=vectorizers,
        ngram_ranges=ngrams,
        samplers=samplers,
        n_iter=25
    )

    results_df.to_csv("data/Master_Grid_Search_Logs.csv", index=False, sep=";")
    print("[+] Grid search complete. Logs saved.")

    # -----------------------------------------------------------------
    # REPORT ASSET GENERATION & EXACT STATE REPLICATION
    # -----------------------------------------------------------------
    print("\n[-->] Executing Report Asset Generation...")
    generator = DataGenerator(results_df=results_df, raw_df=df_mapped, target_col=TargetCol.SENTIMENT)

    # Generate static assets
    generator.generate_stacked_latex_table(output_path="data/task2_preprocessing_examples.tex")
    generator.generate_publication_plots(csv_output="data/task3_evaluation_matrix.csv", tex_output="data/task3_figures.tex")

    # Generate Confusion Matrices using exact configuration reloading
    generator.generate_top_performers_confusion_matrices(
        label_encoder_classes=list(base_pipeline.label_encoder.classes_),
        py_torch_mlp_class=PyTorchMLPClassifier
    )

    # Dynamic Disagreement Analysis using Enums
    print("\n[-->] Dynamically mining the optimal configuration profiles for Error Analysis...")
    target_configs = []

    model_class_map = {
        ModelName.NAIVE_BAYES: MultinomialNB,
        ModelName.FFNN: PyTorchMLPClassifier
    }

    for model_enum, model_class in model_class_map.items():
        model_mask = results_df["Model"] == model_enum
        model_subset = results_df[model_mask]

        if not model_subset.empty:
            best_row = model_subset.loc[model_subset["Macro-F1"].idxmax()]
            parsed_params = ast.literal_eval(best_row["Sampled_Hyperparameters"])
            parsed_ngram = ast.literal_eval(best_row["N-Gram"])

            # Instantiate an exact replica of the winning model
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

if __name__ == "__main__":
    main()
