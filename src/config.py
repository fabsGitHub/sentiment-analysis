import os
import random
import numpy as np
import torch
from enum import Enum

# --- GLOBAL REPRODUCIBILITY ENGINE ---
GLOBAL_SEED = 42

def enforce_reproducibility(seed: int = GLOBAL_SEED):
    """Locks down all pseudo-random number generators across the execution environment."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- SYSTEM ENUMERATIONS ---
class StrategyName(str, Enum):
    STANDARD = "prep_standard"
    FULL = "prep_full"
    MASKED = "prep_masked"
    STANDARD_NUMBERS = "prep_standard_numbers"

class VectorizerName(str, Enum):
    BOW = "BoW"
    TFIDF = "TF-IDF"

class SamplerName(str, Enum):
    NONE = "None (Imbalanced)"
    SMOTE = "SMOTE (Oversampling)"
    SMOTE_TOMEK = "SMOTETomek (Combined)"
    TOMEK = "Tomek (Undersampling)"

class ModelName(str, Enum):
    NAIVE_BAYES = "Naive Bayes"
    FFNN = "FFNN"

class TargetCol(str, Enum):
    SENTIMENT = "sentiment"
