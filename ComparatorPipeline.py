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
