import re
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # <-- Changed to RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy.sparse as sp

# Resampling Imports
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# --- NLTK DOWNLOAD BLOCK ---
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

MONTH_MAP = {
    'jan': '01', 'january': '01', 'feb': '02', 'february': '02',
    'mar': '03', 'march': '03', 'apr': '04', 'april': '04',
    'may': '05', 'jun': '06', 'june': '06',
    'jul': '07', 'july': '07', 'aug': '08', 'august': '08',
    'sep': '09', 'september': '09', 'oct': '10', 'october': '10',
    'nov': '11', 'november': '11', 'dec': '12', 'december': '12'
}

WRITTEN_NUMS = {
    re.compile(r'\bone\b', re.I): '1', re.compile(r'\btwo\b', re.I): '2',
    re.compile(r'\bthree\b', re.I): '3', re.compile(r'\bfour\b', re.I): '4',
    re.compile(r'\bfive\b', re.I): '5', re.compile(r'\bsix\b', re.I): '6',
    re.compile(r'\bseven\b', re.I): '7', re.compile(r'\beight\b', re.I): '8',
    re.compile(r'\bnine\b', re.I): '9', re.compile(r'\bten\b', re.I): '10'
}

FINANCIAL_NOISE_STOPWORDS = {
    '-', "''", "'",
    'year', 'period', 'quarter', 'today', 'first', 'end', 'finnish', 'finland',
    'helsinki', 'hel', 'nokia', 'corporate', 'corporation', 'oyj', 'oy', 'omx', 'group', 'company',
    'said', 'also', 'include', 'including', 'accord', 'according', 'use', 'per', 'part', 'would',
    'base', 'provide'
}

PRESERVED_WORDS = {
    'below', 'but', 'down', 'few', 'more', 'no', 'nor',
    'not', 'only', 'over', 'should', 'up'
}

english_defaults = set(stopwords.words('english'))
CUSTOM_STOPWORDS = (english_defaults | FINANCIAL_NOISE_STOPWORDS) - PRESERVED_WORDS

lemmatizer = WordNetLemmatizer()

PHONE_NUMBER = re.compile(r"(?<!\w)\+[\d\s\-\(\)]{6,20}(?!\w)")
STOCK_TICKER = re.compile(r"\([A-Z]+(\s*:\s*[A-Z0-9]+)?\)")
PHONE_PLACEHOLDER = "__PHONE__"

DATE_RANGE_WITH_YEAR = re.compile(r"\b([a-zA-Z]+)\s+(\d{1,2})\s*-\s*([a-zA-Z]+)\s+(\d{1,2})\s*,?\s*(\d{4})\b", re.I)
MONTH_MONTH_YEAR = re.compile(r"\b([a-zA-Z]+)[-\s]+([a-zA-Z]+)\s+(\d{4})\b", re.I)
MONTH_MONTH = re.compile(r"\b([a-zA-Z]+)[-\s]+([a-zA-Z]+)\b", re.I)
DAY_MONTH_YEAR = re.compile(r"\b(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})\b", re.I)
MONTH_DAY_YEAR = re.compile(r"\b([a-zA-Z]+)\s+(\d{1,2})[,\s]+(\d{4})\b", re.I)
MONTH_YEAR = re.compile(r"\b([a-zA-Z]+)\s+(\d{4})\b", re.I)
YEAR_MONTH = re.compile(r"\b(\d{4})\s+([a-zA-Z]+)\b", re.I)

EARLY_CLEANUP = [
    (re.compile(r"(x[0-9a-fA-F]{4}|[^\x00-\x7F]+)"), " "),
    (re.compile(r"(\d+)(st|nd|rd|th)", re.I), r"\1"),
    (re.compile(r"(\d)\s(\d)"), r"\1\2"),
    (re.compile(r"(\.)\s(\d)"), r"\1\2"),
    (re.compile(r"(\d)\s(\.)"), r"\1\2"),
]

CURRENCIES = r"eur|usd|gbp|jpy|chf|sek|eek"

FINANCIAL_CLEANUP = [
    (re.compile(r"\bus\s*\$", re.I), "usd"),
    (re.compile(r"\beuros?(?=\d)", re.I), "eur "),
    (re.compile(r"\beuros?\b", re.I), "eur"),
    (re.compile(r"\ber\b", re.I), "eur"),
    (re.compile(r"x20ac"), "eur"),
    (re.compile(r"\$"), "usd"),
    (re.compile(r"\%"), "pct"),
    (re.compile(r"\b(\d+\.?\d*)\s*us\s*million\b", re.I), r"\1mn"),
    (re.compile(r"\b(\d+\.?\d*)\s*us\s*m\b", re.I), r"\1mn"),
    (re.compile(r"(\d+\.?\d*)\s*(percent|per cent)", re.I), r"\1pct"),
    (re.compile(r"\bsek\b", re.I), "eur"),
    (re.compile(r"\bmln\b", re.I), "mn"),
    (re.compile(r"\b(\d+\.?\d*)\s*billion\b", re.I), r"\1bn"),
    (re.compile(r"\b(\d+\.?\d*)\s*million\b", re.I), r"\1mn"),
    (re.compile(rf"\b({CURRENCIES})\s*(\d+\.?\d*)\s*\bm\b", re.I), r"\1\2mn"),
    (re.compile(rf"\b(\d+\.?\d*)\s*\bm\s*({CURRENCIES})\b", re.I), r"\2\1mn"),
    (re.compile(rf"\b({CURRENCIES})\s*(\d+\.?\d*)\s*m\b", re.I), r"\1\2mn"),
    (re.compile(rf"({CURRENCIES})\s*([-+]?\d+\.?\d*)\s*(m|mn|bn|k|pct|%)", re.I), r"\1\2\3"),
    (re.compile(r"([-+]?\d+\.?\d*)\s*(m|mn|bn|k|pct|%)", re.I), r"\1\2"),
    (re.compile(rf"({CURRENCIES})\s*([-+]?\d+\.?\d*)", re.I), r"\1\2"),
    (re.compile(rf"([-+]?\d+\.?\d*)\s*({CURRENCIES})(?!\d)", re.I), r"\2\1"),
    (re.compile(rf"\b(\d+\.?\d*)\s*(m|mn|bn|k|pct)\s*({CURRENCIES})\b", re.I), r"\3\1\2"),
    (re.compile(rf"({CURRENCIES})(\d+)\s*,\s*(\d+)\s*(m|mn|bn|k)", re.I), r"\1\2,\3\4"),
    (re.compile(r"(\d+)(pct|mn|bn|k|%)\s*-\s*(\d+)\2", re.I), r"\1-\3\2"),
    (re.compile(r"(\d+),(\d+)"), r"\1.\2"),
]

LATE_CLEANUP = [
    (re.compile(r"(\d{1,2}:\d{2})\s*(am|pm)\b", re.I), r"\1\2"),
    (re.compile(r"\bsq\s*m\b", re.I), "sqm"),
    (re.compile(r"(\d+)\s*(sqm|m|km|kg|g)", re.I), r"\1\2"),
    (re.compile(r"\b([a-zA-Z]+)\s*(\d{1,2})\s*-\s*([a-zA-Z]+)\s*(\d{1,2})\b"), r"\1\2-\3\4"),
    (re.compile(r"(?<!\d)(\d{4})-(\d{2})(?!\d|:)"), r"\1-20\2"),
    (re.compile(r"\b(\d{1,2})-(\d{4})\b"), lambda m: f"{m.group(2)}-{m.group(1).zfill(2)}"),
    (re.compile(r"(?<!\d)[^\w\s'=%-]|[^\w\s'=%-](?!\d)"), ""),
    (re.compile(r"\s+"), " "), (re.compile(r"\s*'(\w+)"), ""),
]

def normalize_dates_smart(text):
    def repl_range_year(m):
        m1, m2 = m.group(1).lower(), m.group(3).lower()
        if m1 in MONTH_MAP and m2 in MONTH_MAP:
            return f"{m.group(5)}-{MONTH_MAP[m1]}-{m.group(2).zfill(2)} to {m.group(5)}-{MONTH_MAP[m2]}-{m.group(4).zfill(2)}"
        return m.group(0)
    text = DATE_RANGE_WITH_YEAR.sub(repl_range_year, text)

    def repl_mo_mo_yr(m):
        m1, m2 = m.group(1).lower(), m.group(2).lower()
        if m1 in MONTH_MAP and m2 in MONTH_MAP:
            return f"{m.group(3)}:{MONTH_MAP[m1]}:{MONTH_MAP[m2]}"
        return m.group(0)
    text = MONTH_MONTH_YEAR.sub(repl_mo_mo_yr, text)

    def repl_mo_mo(m):
        m1, m2 = m.group(1).lower(), m.group(2).lower()
        if m1 in MONTH_MAP and m2 in MONTH_MAP:
            return f"{MONTH_MAP[m1]}:{MONTH_MAP[m2]}"
        return m.group(0)
    text = MONTH_MONTH.sub(repl_mo_mo, text)

    def repl_d_m_y(m):
        mo = m.group(2).lower()
        if mo in MONTH_MAP:
            return f"{m.group(3)}-{MONTH_MAP[mo]}-{m.group(1).zfill(2)}"
        return m.group(0)
    text = DAY_MONTH_YEAR.sub(repl_d_m_y, text)

    def repl_m_d_y(m):
        mo = m.group(1).lower()
        if mo in MONTH_MAP:
            return f"{m.group(3)}-{MONTH_MAP[mo]}-{m.group(2).zfill(2)}"
        return m.group(0)
    text = MONTH_DAY_YEAR.sub(repl_m_d_y, text)

    def repl_m_y(m):
        mo = m.group(1).lower()
        if mo in MONTH_MAP:
            return f"{m.group(2)}:{MONTH_MAP[mo]}"
        return m.group(0)
    text = MONTH_YEAR.sub(repl_m_y, text)

    def repl_y_m(m):
        mo = m.group(2).lower()
        if mo in MONTH_MAP:
            return f"{m.group(1)}:{MONTH_MAP[mo]}"
        return m.group(0)
    text = YEAR_MONTH.sub(repl_y_m, text)
    return text

def _normalize_text_pipeline(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    phones = re.findall(PHONE_NUMBER, text)
    text = PHONE_NUMBER.sub(PHONE_PLACEHOLDER, text)
    text = STOCK_TICKER.sub(lambda m: m.group(0).replace(" ", ""), text)
    for reg, repl in EARLY_CLEANUP: text = reg.sub(repl, text)
    for word_regex, num in WRITTEN_NUMS.items(): text = word_regex.sub(num, text)
    text = normalize_dates_smart(text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    for reg, repl in (FINANCIAL_CLEANUP + LATE_CLEANUP): text = reg.sub(repl, text)
    for p in phones: text = text.replace(PHONE_PLACEHOLDER, p.replace(" ", ""), 1)
    return text

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

# ==========================================
# 2. LEMMATIZATION PREPROCESSING STRATEGIES
# ==========================================
def preprocess_full(text):
    text = _normalize_text_pipeline(text)
    tokens = text.split()
    tagged_tokens = pos_tag(tokens)

    lemmatized_tokens = []
    for word, tag in tagged_tokens:
        word_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=word_pos)
        if lemma not in CUSTOM_STOPWORDS:
            lemmatized_tokens.append(lemma)
    return " ".join(lemmatized_tokens).strip()

def preprocess_masked(text):
    text = _normalize_text_pipeline(text)
    text = re.sub(r'\+[\d\-()]{6,20}', ' [PHONE] ', text)
    text = re.sub(r'\b(eur|usd|gbp|jpy|chf|sek|eek)\d+\.?\d*(mn|bn|k|pct)?\b', ' [MONEY] ', text)
    text = re.sub(r'\b\d+\.?\d*pct\b', ' [PERCENT] ', text)
    text = re.sub(r'\b\d{4}:\d{2}:\d{2}\b', ' [DATE] ', text)
    text = re.sub(r'\b\d{4}:\d{2}\b', ' [DATE] ', text)
    text = re.sub(r'\b\d{2}:\d{2}\b', ' [DATE] ', text)
    text = re.sub(r'\b(19\d{2}|20\d{2})\b', ' [DATE] ', text)
    text = re.sub(r'\b\d{1,2}:\d{2}(am|pm)\b', ' [TIME] ', text)
    text = re.sub(r'\b\d+\.?\d*(sqm|km|kg|m|g)\b', ' [MEASUREMENT] ', text)
    text = re.sub(r'\b\d+\.?\d*\b', ' [NUMBER] ', text)

    tokens = text.split()
    tagged_tokens = pos_tag(tokens)

    mask_placeholders = {
        '[PHONE]', '[MONEY]', '[PERCENT]',
        '[DATE]', '[TIME]', '[MEASUREMENT]', '[NUMBER]'
    }
    processed_tokens = []
    for word, tag in tagged_tokens:
        if word in mask_placeholders:
            processed_tokens.append(word)
        else:
            word_pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, pos=word_pos)
            if lemma not in CUSTOM_STOPWORDS:
                processed_tokens.append(lemma)
    return re.sub(r'\s+', ' ', " ".join(processed_tokens)).strip()

def preprocess_standard_optimized(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tagged_tokens = pos_tag(tokens)

    processed_tokens = []
    for word, tag in tagged_tokens:
        word_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=word_pos)
        if lemma not in CUSTOM_STOPWORDS:
            processed_tokens.append(lemma)
    return " ".join(processed_tokens).strip()

def preprocess_standard(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^[a-zA-Z]\s]', ' ', text)
    tokens = text.split()
    tagged_tokens = pos_tag(tokens)

    processed_tokens = []
    for word, tag in tagged_tokens:
        word_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=word_pos)
        if lemma not in CUSTOM_STOPWORDS:
            processed_tokens.append(lemma)
    return " ".join(processed_tokens).strip()

# ==========================================
# 3. EVALUATION & MODEL METRICS
# ==========================================
def evaluate_model(y_true, y_pred, model_display_name):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return acc, macro_f1

# ==========================================
# 4. HIGH-PERFORMANCE PYTORCH MLP BACKEND
# ==========================================
class DenseTextDataset(Dataset):
    """Expects pre-converted dense numpy/torch arrays to completely avoid inline CPU overhead."""
    def __init__(self, X, y=None):
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X.float() if torch.is_tensor(X) else torch.tensor(X).float()

        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(64,), activation='relu', solver='adam',
                 alpha=0.0001, batch_size=128, learning_rate_init=0.001,
                 max_iter=20, early_stopping=True, validation_fraction=0.1, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Handle sparse to dense conversion ONCE per fit lifecycle here safely
        if sp.issparse(X):
            X = X.toarray()

        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        input_dim = X.shape[1]

        layers = []
        prev_dim = input_dim
        hidden_sizes = self.hidden_layer_sizes if isinstance(self.hidden_layer_sizes, tuple) else (self.hidden_layer_sizes,)

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.activation == 'relu': layers.append(nn.ReLU())
            elif self.activation == 'tanh': layers.append(nn.Tanh())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers).to(self.device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        dataset = DenseTextDataset(X, np.array(y))

        if self.early_stopping and self.validation_fraction > 0:
            val_size = int(len(dataset) * self.validation_fraction)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            train_dataset = dataset
            val_loader = None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        epochs_no_improve = 0
        patience = 2

        for epoch in range(self.max_iter):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_x)
                        val_loss += criterion(outputs, batch_y).item()
                val_loss /= len(val_loader)

                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break
        return self

    def predict(self, X):
        if sp.issparse(X):
            X = X.toarray()
        self.model.eval()
        dataset = DenseTextDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        return np.array(predictions)

# ==========================================
# 5. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    TARGET_COL = "sentiment"
    try:
        df = pd.read_csv("Sentences_50Agree.txt", sep="@", header=None, names=["sentence", "sentiment"])
    except FileNotFoundError:
        print("Error: 'Sentences_50Agree.txt' not found.")
        exit()

    label_encoder = LabelEncoder()
    df[TARGET_COL] = label_encoder.fit_transform(df[TARGET_COL])
    classes_multiclass = label_encoder.classes_

    print("Running text preprocessing routines...")
    df["prep_standard"] = df["sentence"].apply(preprocess_standard)
    df["prep_full"] = df["sentence"].apply(preprocess_full)
    df["prep_masked"] = df["sentence"].apply(preprocess_masked)
    df["prep_standard_numbers"] = df["sentence"].apply(preprocess_standard_optimized)

    strategies = ["prep_standard", "prep_full", "prep_masked", "prep_standard_numbers"]
    samplers = {
        "None (Imbalanced)": None,
        "SMOTE (Oversampling)": SMOTE(random_state=42),
        "SMOTETomek (Combined)": SMOTETomek(random_state=42)
    }

    custom_token_pattern = r'(?u)\[?\b\w[-\w\.]*\b\]?'
    master_results = []

    for strategy in strategies:
        print(f"\nProcessing Strategy Column: {strategy}")
        df[strategy] = df[strategy].replace(r'^\s*$', np.nan, regex=True)
        df_clean = df.dropna(subset=[strategy, TARGET_COL]).reset_index(drop=True)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            df_clean[strategy], df_clean[TARGET_COL], test_size=0.20, random_state=42, stratify=df_clean[TARGET_COL]
        )

        # FIX 1: Max features capping limits feature dimensions and prevents CPU memory thrashing
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), token_pattern=custom_token_pattern, max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train_raw)
        X_test_vec = vectorizer.transform(X_test_raw)

        for sampler_name, sampler in samplers.items():
            print(f"  -> Sampler: {sampler_name}")

            if sampler is not None:
                X_train_res, y_train_res = sampler.fit_resample(X_train_vec, y_train)
            else:
                X_train_res, y_train_res = X_train_vec, y_train

            # FIX 2: Broadened hyperparameter distributions to run inside an efficient random search layout
            param_dist = {
                'hidden_layer_sizes': [(32,), (64,),(64, 32) ],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'batch_size': [64,128, 256],
                'learning_rate_init': [0.001, 0.005, 0.01],
                'max_iter': [100] # Kept small since early stopping handles local convergence
            }

            pytorch_mlp = PyTorchMLPClassifier(random_state=42)

            # FIX 3: RandomizedSearchCV checks 8 random high-performing combinations instead of 288
            search_engine = RandomizedSearchCV(
                estimator=pytorch_mlp,
                param_distributions=param_dist,
                n_iter=12,
                cv=2,
                scoring='f1_macro',
                n_jobs=1,
                random_state=42,
                verbose=0
            )

            search_engine.fit(X_train_res, y_train_res)

            best_model = search_engine.best_estimator_
            y_pred = best_model.predict(X_test_vec)
            acc, macro_f1 = evaluate_model(y_test, y_pred, f"{strategy} + {sampler_name}")

            master_results.append({
                "Strategy": strategy,
                "Sampling": sampler_name,
                "Best Activation": search_engine.best_params_['activation'],
                "Best Hidden Layers": search_engine.best_params_['hidden_layer_sizes'],
                "Test Accuracy": acc,
                "Test Macro F1": macro_f1
            })

            del X_train_res, y_train_res
            gc.collect()

    df_results = pd.DataFrame(master_results)
    print("\n" + "="*50 + "\nGLOBAL BENCHMARK RESULTS\n" + "="*50)
    print(df_results.sort_values(by="Test Macro F1", ascending=False).to_string(index=False))
    df_results.to_csv("sentiment_analysis_results.csv", index=False)
