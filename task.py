import re
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.combine import SMOTETomek
import scipy.sparse as sp

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ==========================================
# --- 1. Configurations & Maps ---
# ==========================================

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
    '-', "''", "'", '2009', '2008', '2007', '2006', '2010', '2005', '1', '2',
    'year', 'period', 'quarter', 'today', 'first', 'end',
    'finnish', 'finland', 'helsinki', 'hel', 'nokia',
    'oyj', 'oy', 'corpor', 'omx', 'group', 'compani',
    'said', 'also', 'includ', 'accord', 'use', 'well',
    'per', 'part', 'would', 'base', 'provid'
}

PRESERVED_WORDS = {
    'good', 'bad', 'high', 'low', 'risk', 'profit', 'loss',
    'up', 'down', 'increase', 'decrease', 'increas', 'decreas',
    'strong', 'weak', 'better', 'worse', 'positive', 'negative',
    'stable', 'volatile', 'only', 'below', 'few', 'more',
    'no', 'not', 'nor', 'over', 'should', 'but'
}

english_defaults = set(stopwords.words('english'))
CUSTOM_STOPWORDS = (english_defaults | FINANCIAL_NOISE_STOPWORDS) - PRESERVED_WORDS
stemmer = PorterStemmer()

PHONE_NUMBER = re.compile(r"(?<!\w)\+[\d\s\-\(\)]{6,20}(?!\w)")
STOCK_TICKER = re.compile(r"\([A-Z]+(\s*:\s*[A-Z0-9]+)?\)")
PHONE_PLACEHOLDER = "__PHONE__"

DATE_RANGE_WITH_YEAR = re.compile(r"\b([a-zA-Z]+)\s+(\d{1,2})\s*-\s*([a-zA-Z]+)\s+(\d{1,2})\s*,?\s*(\d{4})\b", re.I)
MONTH_MONTH_YEAR = re.compile(r"\b([a-zA-Z]+)-([a-zA-Z]+)\s+(\d{4})\b", re.I)
MONTH_MONTH = re.compile(r"\b([a-zA-Z]+)-([a-zA-Z]+)\b", re.I)
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

FINANCIAL_CLEANUP = [
    (re.compile(r"x20ac"), "eur"), (re.compile(r"\$"), "usd"), (re.compile(r"\%"), "pct"),
    (re.compile(r"(\d+\.?\d*)\s*(percent|per cent)"), r"\1pct"),
    (re.compile(r"\b(euros?|SEK|sek)\b", re.I), "eur"),
    (re.compile(r"\bmln\b", re.I), "mn"),
    (re.compile(r"\b(\d+\.?\d*)\s*billion\b", re.I), r"\1bn"),
    (re.compile(r"\b(\d+\.?\d*)\s*million\b", re.I), r"\1mn"),
    (re.compile(r"(eur|usd|gbp|jpy|chf|sek)\s*([-+]?\d+\.?\d*)\s*(m|mn|bn|k|pct|%)", re.I), r"\1\2\3"),
    (re.compile(r"([-+]?\d+\.?\d*)\s*(m|mn|bn|k|pct|%)", re.I), r"\1\2"),
    (re.compile(r"(eur|usd|gbp|jpy|chf|sek)\s*([-+]?\d+\.?\d*)", re.I), r"\1\2"),
    (re.compile(r"([-+]?\d+\.?\d*)\s*(eur|usd|gbp|jpy|chf|sek|gmt)(?!\d)", re.I), r"\2\1"),
    (re.compile(r"(\d+\.?\d*)\s*(m|mn|bn|k|pct)\s*(eur|usd|gbp|jpy|chf|sek)", re.I), r"\3\1\2"),
    (re.compile(r"(eur|usd|gbp|jpy|chf|sek)(\d+)\s*,\s*(\d+)\s*(m|mn|bn|k)", re.I), r"\1\2,\3\4"),
    (re.compile(r"(\d+)(pct|mn|bn|k|%)\s*-\s*(\d+)\2", re.I), r"\1-\3\2"),
    (re.compile(r"(\d+),(\d+)"), r"\1.\2"),
]

LATE_CLEANUP = [
    (re.compile(r"(\d{1,2}:\d{2})\s*(am|pm)\b", re.I), r"\1\2"),
    (re.compile(r"\bsq\s*m\b", re.I), "sqm"),
    (re.compile(r"(\d+)\s*(sqm|m|km|kg|g)", re.I), r"\1\2"),
    (re.compile(r"\b([a-zA-Z]+)\s*(\d{1,2})\s*-\s*([a-zA-Z]+)\s*(\d{1,2})\b"), r"\1\2-\3\4"),
    (re.compile(r"(?<!\d)(\d{4})-(\d{2})(?!\d)"), r"\1-20\2"),
    (re.compile(r"\b(\d{1,2})-(\d{4})\b"), lambda m: f"{m.group(2)}-{m.group(1).zfill(2)}"),
    (re.compile(r"(?<!\d)[^\w\s'=%-]|[^\w\s'=%-](?!\d)"), ""),
    (re.compile(r"\s+"), " "), (re.compile(r"\s*'(\w+)"), ""),
]

# ==========================================
# --- 2. Preprocessing Logic ---
# ==========================================

def normalize_dates_smart(text):
    def repl_range_year(m):
        m1, m2 = m.group(1).lower(), m.group(3).lower()
        if m1 in MONTH_MAP and m2 in MONTH_MAP: return f"{m.group(5)}-{MONTH_MAP[m1]}-{m.group(2).zfill(2)} to {m.group(5)}-{MONTH_MAP[m2]}-{m.group(4).zfill(2)}"
        return m.group(0)
    text = DATE_RANGE_WITH_YEAR.sub(repl_range_year, text)

    def repl_mo_mo_yr(m):
        m1, m2 = m.group(1).lower(), m.group(2).lower()
        if m1 in MONTH_MAP and m2 in MONTH_MAP: return f"{m.group(3)}-{MONTH_MAP[m1]}-{MONTH_MAP[m2]}"
        return m.group(0)
    text = MONTH_MONTH_YEAR.sub(repl_mo_mo_yr, text)

    def repl_mo_mo(m):
        m1, m2 = m.group(1).lower(), m.group(2).lower()
        if m1 in MONTH_MAP and m2 in MONTH_MAP: return f"{MONTH_MAP[m1]}-{MONTH_MAP[m2]}"
        return m.group(0)
    text = MONTH_MONTH.sub(repl_mo_mo, text)

    def repl_d_m_y(m):
        mo = m.group(2).lower()
        if mo in MONTH_MAP: return f"{m.group(3)}-{MONTH_MAP[mo]}-{m.group(1).zfill(2)}"
        return m.group(0)
    text = DAY_MONTH_YEAR.sub(repl_d_m_y, text)

    def repl_m_d_y(m):
        mo = m.group(1).lower()
        if mo in MONTH_MAP: return f"{m.group(3)}-{MONTH_MAP[mo]}-{m.group(2).zfill(2)}"
        return m.group(0)
    text = MONTH_DAY_YEAR.sub(repl_m_d_y, text)

    def repl_m_y(m):
        mo = m.group(1).lower()
        if mo in MONTH_MAP: return f"{m.group(2)}-{MONTH_MAP[mo]}"
        return m.group(0)
    text = MONTH_YEAR.sub(repl_m_y, text)

    def repl_y_m(m):
        mo = m.group(2).lower()
        if mo in MONTH_MAP: return f"{m.group(1)}-{MONTH_MAP[mo]}"
        return m.group(0)
    return YEAR_MONTH.sub(repl_y_m, text)

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

def preprocess_full(text):
    text = _normalize_text_pipeline(text)
    tokens = [word for word in text.split() if word not in CUSTOM_STOPWORDS]
    return " ".join([stemmer.stem(word) for word in tokens]).strip()

def preprocess_masked(text):
    text = _normalize_text_pipeline(text)
    text = re.sub(r'\+[\d\-()]{6,20}', ' [PHONE_MARKER] ', text)
    text = re.sub(r'\b(eur|usd|gbp|jpy|chf|sek)\d+\.?\d*(mn|bn|k|pct)?\b', ' [MONEY_METRIC] ', text)
    text = re.sub(r'\b\d+\.?\d*pct\b', ' [PERCENT_METRIC] ', text)
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', ' [DATE_MARKER] ', text)
    text = re.sub(r'\b\d{4}-\d{2}\b', ' [DATE_MARKER] ', text)
    text = re.sub(r'\b\d{2}-\d{2}\b', ' [DATE_MARKER] ', text)
    text = re.sub(r'\b(19\d{2}|20\d{2})\b', ' [DATE_MARKER] ', text)
    text = re.sub(r'\b\d{1,2}:\d{2}(am|pm)\b', ' [TIME_MARKER] ', text)
    text = re.sub(r'\b\d+\.?\d*(sqm|km|kg|m|g)\b', ' [MEASUREMENT_MARKER] ', text)
    text = re.sub(r'\b\d+\.?\d*\b', ' [NUMBER_MARKER] ', text)

    tokens = text.split()
    processed_tokens = []
    mask_placeholders = {'[PHONE_MARKER]', '[MONEY_METRIC]', '[PERCENT_METRIC]', '[DATE_MARKER]', '[TIME_MARKER]', '[MEASUREMENT_MARKER]', '[NUMBER_MARKER]'}

    for word in tokens:
        if word in mask_placeholders: processed_tokens.append(word)
        elif word not in CUSTOM_STOPWORDS: processed_tokens.append(stemmer.stem(word))

    return re.sub(r'\s+', ' ', " ".join(processed_tokens)).strip()

def preprocess_standard(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return " ".join([stemmer.stem(word) for word in text.split() if word not in CUSTOM_STOPWORDS]).strip()

def evaluate_model(y_true, y_pred, model_display_name, classes):
    print(f"\n================ {model_display_name} Evaluation ================")
    print(classification_report(y_true, y_pred, target_names=classes))
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f} | Macro F1-Score: {macro_f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'CM: {model_display_name}', fontsize=10)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

# ==========================================
# --- 3. GPU-Optimized PyTorch Backend ---
# ==========================================

class SparseTextDataset(Dataset):
    """Custom Dataset to prevent OOM RAM errors by lazily converting sparse rows to dense."""
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self.is_sparse = sp.issparse(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Fetch row and convert to dense only when needed by the DataLoader batch
        x_row = self.X[idx]
        if self.is_sparse:
            x_row = x_row.toarray().flatten()

        x_tensor = torch.tensor(x_row, dtype=torch.float32)

        if self.y is not None:
            return x_tensor, torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor

class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(64,), activation='relu', solver='adam',
                 alpha=0.0001, batch_size=32, learning_rate_init=0.001,
                 max_iter=50, early_stopping=True, validation_fraction=0.1, random_state=42):
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
        self.pin_memory = True if torch.cuda.is_available() else False

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        input_dim = X.shape[1]

        # Build Standard Model Layout
        layers = []
        prev_dim = input_dim
        # Check if single int was passed instead of tuple via grid search
        hidden_sizes = self.hidden_layer_sizes if isinstance(self.hidden_layer_sizes, tuple) else (self.hidden_layer_sizes,)

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers).to(self.device)
        criterion = nn.CrossEntropyLoss()

        if self.solver == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)

        # Wrap in memory-safe Custom Dataset
        dataset = SparseTextDataset(X, np.array(y))

        if self.early_stopping and self.validation_fraction > 0:
            val_size = int(len(dataset) * self.validation_fraction)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory)
        else:
            train_dataset = dataset
            val_loader = None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory)

        best_loss = float('inf')
        epochs_no_improve = 0
        patience = 5

        for epoch in range(self.max_iter):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True)
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
                        batch_x, batch_y = batch_x.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True)
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

        # Free up computational graph VRAM explicitly after fitting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return self

    def predict(self, X):
        self.model.eval()
        dataset = SparseTextDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory)

        predictions = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        # Clear VRAM Post-Prediction to protect memory during large GridSearches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.array(predictions)

# ==========================================
# --- 4. Main Execution Pipeline ---
# ==========================================

if __name__ == "__main__":
    TARGET_COL = "sentiment"
    # Ensure correct file path here!
    try:
        df = pd.read_csv("Sentences_50Agree.txt", sep="@", header=None, names=["sentence", "sentiment"])
    except FileNotFoundError:
        print("Error: 'Sentences_50Agree.txt' not found. Please ensure the dataset is in the working directory.")
        exit()

    label_encoder = LabelEncoder()
    df[TARGET_COL] = label_encoder.fit_transform(df[TARGET_COL])
    classes_multiclass = label_encoder.classes_

    df["prep_standard"] = df["sentence"].apply(preprocess_standard)
    df["prep_full"] = df["sentence"].apply(preprocess_full)
    df["prep_masked"] = df["sentence"].apply(preprocess_masked)

    ngram_ranges = [(1, 2)] # Reduced list for sane execution time
    strategies = ["prep_standard", "prep_full", "prep_masked"]
    custom_token_pattern = r'(?u)\[?\b\w[-\w\.]*\b\]?'

    results = []
    feature_sizes = []

    models = {
        "Naive Bayes": MultinomialNB(),
        "FFNN (Baseline)": PyTorchMLPClassifier(
            hidden_layer_sizes=(64,), activation='relu', solver='adam',
            max_iter=50, early_stopping=True, random_state=42
        )
    }

    active_device = "GPU Acceleration (CUDA)" if torch.cuda.is_available() else "CPU Execution Loop"
    print(f"Starting Evaluation Loop on: {active_device}. This may take a few minutes...")

    for strategy in strategies:
        print(f"\n---> Processing Strategy: {strategy.upper()}")
        df_clean = df.dropna(subset=[strategy, TARGET_COL]).reset_index(drop=True)
        X_train, X_test, y_train, y_test = train_test_split(
            df_clean[strategy], df_clean[TARGET_COL],
            test_size=0.20, random_state=42, stratify=df_clean[TARGET_COL]
        )

        smote_tomek = SMOTETomek(random_state=42)

        for ngram in ngram_ranges:
            print(f"     N-Gram: {ngram}")

            # --- BoW ---
            count_vect = CountVectorizer(ngram_range=ngram, token_pattern=custom_token_pattern)
            X_train_bow_raw = count_vect.fit_transform(X_train)
            X_test_bow = count_vect.transform(X_test)
            X_train_bow, y_train_resampled_bow = smote_tomek.fit_resample(X_train_bow_raw, y_train)

            feature_sizes.append({"Strategy": strategy, "Vectorizer": "BoW", "N-Gram": str(ngram), "Feature_Size": len(count_vect.get_feature_names_out())})

            # --- TF-IDF ---
            tfidf_vect = TfidfVectorizer(ngram_range=ngram, token_pattern=custom_token_pattern)
            X_train_tfidf_raw = tfidf_vect.fit_transform(X_train)
            X_test_tfidf = tfidf_vect.transform(X_test)
            X_train_tfidf, y_train_resampled_tfidf = smote_tomek.fit_resample(X_train_tfidf_raw, y_train)

            feature_sizes.append({"Strategy": strategy, "Vectorizer": "TF-IDF", "N-Gram": str(ngram), "Feature_Size": len(tfidf_vect.get_feature_names_out())})

            for model_name, model_instance in models.items():
                # Evaluate BoW
                model_instance.fit(X_train_bow, y_train_resampled_bow)
                y_pred_bow = model_instance.predict(X_test_bow)
                results.append({
                    "Model": model_name, "Strategy": strategy, "Vectorizer": "BoW",
                    "N-Gram": str(ngram), "Accuracy": accuracy_score(y_test, y_pred_bow),
                    "Macro-F1": f1_score(y_test, y_pred_bow, average='macro')
                })

                # Evaluate TF-IDF
                model_instance.fit(X_train_tfidf, y_train_resampled_tfidf)
                y_pred_tfidf = model_instance.predict(X_test_tfidf)
                results.append({
                    "Model": model_name, "Strategy": strategy, "Vectorizer": "TF-IDF",
                    "N-Gram": str(ngram), "Accuracy": accuracy_score(y_test, y_pred_tfidf),
                    "Macro-F1": f1_score(y_test, y_pred_tfidf, average='macro')
                })

    results_df = pd.DataFrame(results)
    print("\nEvaluation Complete!")

    big_matrix = pd.pivot_table(results_df, values='Macro-F1', index=['Model', 'Vectorizer', 'N-Gram'], columns=['Strategy']).loc[:, ['prep_standard', 'prep_full', 'prep_masked']]
    print("\n========================= THE MASTER PERFORMANCE MATRIX (Macro-F1) =========================")
    print(big_matrix.round(4))

    sns.set_theme(style="whitegrid")
    g = sns.catplot(data=results_df, x='Strategy', y='Macro-F1', hue='N-Gram', col='Vectorizer', row='Model', kind='bar', palette="viridis", height=4, aspect=1.5)
    g.fig.suptitle("Global Comparison", y=1.02, fontsize=16)
    plt.show()

    # ==========================================
    # --- 5. Grid Search Validation ---
    # ==========================================
    print("\n========== PHASE 2: GRID SEARCH ON THE WINNING COMBINATION ==========")
    BEST_STRATEGY = "prep_masked"
    BEST_NGRAM = (1, 2)

    df_winner = df.dropna(subset=[BEST_STRATEGY, TARGET_COL]).reset_index(drop=True)
    X_train_win, X_test_win, y_train_win, y_test_win = train_test_split(
        df_winner[BEST_STRATEGY], df_winner[TARGET_COL], test_size=0.20, random_state=42, stratify=df_winner[TARGET_COL]
    )

    final_vect = TfidfVectorizer(ngram_range=BEST_NGRAM, token_pattern=custom_token_pattern)
    X_train_final = final_vect.fit_transform(X_train_win)
    X_test_final = final_vect.transform(X_test_win)

    # Simplified Grid Search for demonstration limits (Adjust as needed)
    param_grid = {
    'hidden_layer_sizes': [(32,), (64,), (32, 32), (64, 32), (64,64), (128,)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [20, 50, 100],
    'early_stopping': [True],
    'validation_fraction': [0.1],
    'random_state': [42]
}

    pytorch_mlp = PyTorchMLPClassifier()
    grid_search = GridSearchCV(estimator=pytorch_mlp, param_grid=param_grid, cv=2, scoring='f1_macro', n_jobs=1, verbose=2)

    print(f"Executing Grid Search on {X_train_final.shape[0]} samples and {X_train_final.shape[1]} features...")
    grid_search.fit(X_train_final, y_train_win)

    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    y_pred_mlp = best_model.predict(X_test_final)

    evaluate_model(y_test_win, y_pred_mlp, f"Tuned PyTorch FFNN ({BEST_STRATEGY}, TF-IDF, {BEST_NGRAM})", classes_multiclass)
