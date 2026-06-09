import abc
import re
from typing import Set, Dict, Pattern, List, Tuple, Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from config import StrategyName

# ==========================================
# INTERFACE & STRATEGY SPECIFICATION
# ==========================================

class BasePreprocessorStrategy(abc.ABC):
    """Abstract interface defining the execution protocol for text preprocessing strategies."""

    @abc.abstractmethod
    def preprocess(self, text: str) -> str:
        """
        Execute core text transformation sequence.

        :param text: Raw input string.
        :return: Normalized, transformed, and tokenized/lemmatized string.
        """
        pass


# ==========================================
# PREPROCESSOR CONTROLLER / CORE CONTEXT
# ==========================================

class Preprocessor:
    """
    Unified entry-point and configuration engine for text preprocessing.
    Handles global initializations, asset downloads, and delegates processing
    sub-routines to structural execution strategies.
    """

    # --- Class-level Constants & Preprocessing Maps ---
    MONTH_MAP: Dict[str, str] = {
        'jan': '01', 'january': '01', 'feb': '02', 'february': '02',
        'mar': '03', 'march': '03', 'apr': '04', 'april': '04',
        'may': '05', 'jun': '06', 'june': '06',
        'jul': '07', 'july': '07', 'aug': '08', 'august': '08',
        'sep': '09', 'september': '09', 'oct': '10', 'october': '10',
        'nov': '11', 'november': '11', 'dec': '12', 'december': '12'
    }

    WRITTEN_NUMS: Dict[Pattern, str] = {
        re.compile(r'\bone\b', re.I): '1', re.compile(r'\btwo\b', re.I): '2',
        re.compile(r'\bthree\b', re.I): '3', re.compile(r'\bfour\b', re.I): '4',
        re.compile(r'\bfive\b', re.I): '5', re.compile(r'\bsix\b', re.I): '6',
        re.compile(r'\bseven\b', re.I): '7', re.compile(r'\beight\b', re.I): '8',
        re.compile(r'\bnine\b', re.I): '9', re.compile(r'\bten\b', re.I): '10'
    }

    FINANCIAL_NOISE_STOPWORDS: Set[str] = {
        '-', "''", "'", 'year', 'period', 'quarter', 'today', 'first', 'end',
        'finnish', 'finland', 'helsinki', 'hel', 'nokia', 'corporate',
        'corporation', 'oyj', 'oy', 'omx', 'group', 'company', 'said', 'also',
        'include', 'including', 'accord', 'according', 'use', 'per', 'part',
        'would', 'base', 'provide'
    }

    PRESERVED_WORDS: Set[str] = {
        'below', 'but', 'down', 'few', 'more', 'no', 'nor', 'not', 'only',
        'over', 'should', 'up'
    }

    # Regex Foundations
    PHONE_NUMBER: Pattern = re.compile(r"(?<!\w)\+[\d\s\-\(\)]{6,20}(?!\w)")
    STOCK_TICKER: Pattern = re.compile(r"\([A-Z]+(\s*:\s*[A-Z0-9]+)?\)")
    PHONE_PLACEHOLDER: str = "__PHONE__"

    DATE_RANGE_WITH_YEAR: Pattern = re.compile(r"\b([a-zA-Z]+)\s+(\d{1,2})\s*-\s*([a-zA-Z]+)\s+(\d{1,2})\s*,?\s*(\d{4})\b", re.I)
    MONTH_MONTH_YEAR: Pattern = re.compile(r"\b([a-zA-Z]+)[-\s]+([a-zA-Z]+)\s+(\d{4})\b", re.I)
    MONTH_MONTH: Pattern = re.compile(r"\b([a-zA-Z]+)[-\s]+([a-zA-Z]+)\b", re.I)
    DAY_MONTH_YEAR: Pattern = re.compile(r"\b(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})\b", re.I)
    MONTH_DAY_YEAR: Pattern = re.compile(r"\b([a-zA-Z]+)\s+(\d{1,2})[,\s]+(\d{4})\b", re.I)
    MONTH_YEAR: Pattern = re.compile(r"\b([a-zA-Z]+)\s+(\d{4})\b", re.I)
    YEAR_MONTH: Pattern = re.compile(r"\b(\d{4})\s+([a-zA-Z]+)\b", re.I)

    EARLY_CLEANUP: List[Tuple[Pattern, str]] = [
        (re.compile(r"(x[0-9a-fA-F]{4}|[^\x00-\x7F]+)"), " "),
        (re.compile(r"(\d+)(st|nd|rd|th)", re.I), r"\1"),
        (re.compile(r"(\d)\s(\d)"), r"\1\2"),
        (re.compile(r"(\.)\s(\d)"), r"\1\2"),
        (re.compile(r"(\d)\s(\.)"), r"\1\2"),
    ]

    CURRENCIES: str = r"eur|usd|gbp|jpy|chf|sek|eek"

    FINANCIAL_CLEANUP: List[Tuple[Pattern, str]] = [
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
        (re.compile(r"({CURRENCIES})\s*([-+]?\d+\.?\d*)\s*(m|mn|bn|k|pct|%)", re.I), r"\1\2\3"),
        (re.compile(r"([-+]?\d+\.?\d*)\s*(m|mn|bn|k|pct|%)", re.I), r"\1\2"),
        (re.compile(rf"({CURRENCIES})\s*([-+]?\d+\.?\d*)", re.I), r"\1\2"),
        (re.compile(rf"([-+]?\d+\.?\d*)\s*({CURRENCIES})(?!\d)", re.I), r"\2\1"),
        (re.compile(rf"\b(\d+\.?\d*)\s*(m|mn|bn|k|pct)\s*({CURRENCIES})\b", re.I), r"\3\1\2"),
        (re.compile(rf"({CURRENCIES})(\d+)\s*,\s*(\d+)\s*(m|mn|bn|k)", re.I), r"\1\2,\3\4"),
        (re.compile(r"(\d+)(pct|mn|bn|k|%)\s*-\s*(\d+)\2", re.I), r"\1-\3\2"),
        (re.compile(r"(\d+),(\d+)"), r"\1.\2"),
    ]

    LATE_CLEANUP: List[Tuple[Pattern, str]] = [
        (re.compile(r"(\d{1,2}:\d{2})\s*(am|pm)\b", re.I), r"\1\2"),
        (re.compile(r"\bsq\s*m\b", re.I), "sqm"),
        (re.compile(r"(\d+)\s*(sqm|m|km|kg|g)", re.I), r"\1\2"),
        (re.compile(r"\b([a-zA-Z]+)\s*(\d{1,2})\s*-\s*([a-zA-Z]+)\s*(\d{1,2})\b"), r"\1\2-\3\4"),
        (re.compile(r"(?<!\d)(\d{4})-(\d{2})(?!\d|:)"), r"\1-20\2"),
        (re.compile(r"\b(\d{1,2})-(\d{4})\b"), lambda m: f"{m.group(2)}-{m.group(1).zfill(2)}"),
        (re.compile(r"\s+"), " "), (re.compile(r"\s*'(\w+)"), ""),
        (re.compile(r"\s+"), " "),
        (re.compile(r"\s*'(\w+)"), ""),
    ]



    def preprocess(self, text: str, strategy: Any) -> str:
        """
        Unified entry-point that resolves string strategies
        and applies the correct sub-routine class.
        """
        if isinstance(strategy, str):
            strategy = strategy.lower().replace("prep_", "")

        # Look up names from globals dynamically to avoid order-of-definition or import traps
        strategy_classes = {
            "standard": "StandardStrategy",
            "full": "FullStrategy",
            "masked": "MaskedStrategy",
            "standard_numbers": "StandardNumbersStrategy"
        }

        class_name = strategy_classes.get(strategy, "StandardStrategy")

        # Pull class definition safely from the module environment
        strategy_class = globals().get(class_name)
        if strategy_class is None:
            raise AttributeError(f"Strategy class '{class_name}' could not be resolved in Preprocessor module scope.")

        worker = strategy_class(self)
        return worker.preprocess(text)

    def __init__(self, download_resources: bool = True):
        """Initializes internal NLP instances and state configurations."""
        if download_resources:
            self._bootstrap_nltk()

        self.lemmatizer = WordNetLemmatizer()

        # Build derived custom configurations
        english_defaults = set(stopwords.words('english'))
        self.custom_stopwords = (english_defaults | self.FINANCIAL_NOISE_STOPWORDS) - self.PRESERVED_WORDS

        # Initialize internal strategies mapping
        self._strategies: Dict[str, BasePreprocessorStrategy] = {
            'full': FullPreprocessingStrategy(self),
            'masked': MaskedPreprocessingStrategy(self),
            'standard_optimized': StandardOptimizedStrategy(self),
            'standard': StandardStrategy(self)
        }

    @staticmethod
    def _bootstrap_nltk():
        """Handles background quiet asset updates for NLTK dependency trees."""
        for pkg in ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
            nltk.download(pkg, quiet=True)

    @staticmethod
    def get_wordnet_pos(treebank_tag: str) -> str:
        """Maps default POS tags directly to the format expected by the Lemmatizer."""
        if treebank_tag.startswith('J'): return wordnet.ADJ
        elif treebank_tag.startswith('V'): return wordnet.VERB
        elif treebank_tag.startswith('N'): return wordnet.NOUN
        elif treebank_tag.startswith('R'): return wordnet.ADV
        return wordnet.NOUN

    def normalize_dates_smart(self, text: str) -> str:
        """Normalizes variant textual variations of dates into ISO-esque patterns."""
        def repl_range_year(m):
            m1, m2 = m.group(1).lower(), m.group(3).lower()
            if m1 in self.MONTH_MAP and m2 in self.MONTH_MAP:
                return f"{m.group(5)}-{self.MONTH_MAP[m1]}-{m.group(2).zfill(2)} to {m.group(5)}-{self.MONTH_MAP[m2]}-{m.group(4).zfill(2)}"
            return m.group(0)
        text = self.DATE_RANGE_WITH_YEAR.sub(repl_range_year, text)

        def repl_mo_mo_yr(m):
            m1, m2 = m.group(1).lower(), m.group(2).lower()
            if m1 in self.MONTH_MAP and m2 in self.MONTH_MAP:
                return f"{m.group(3)}:{self.MONTH_MAP[m1]}:{self.MONTH_MAP[m2]}"
            return m.group(0)
        text = self.MONTH_MONTH_YEAR.sub(repl_mo_mo_yr, text)

        def repl_mo_mo(m):
            m1, m2 = m.group(1).lower(), m.group(2).lower()
            if m1 in self.MONTH_MAP and m2 in self.MONTH_MAP:
                return f"{self.MONTH_MAP[m1]}:{self.MONTH_MAP[m2]}"
            return m.group(0)
        text = self.MONTH_MONTH.sub(repl_mo_mo, text)

        def repl_d_m_y(m):
            mo = m.group(2).lower()
            if mo in self.MONTH_MAP:
                return f"{m.group(3)}-{self.MONTH_MAP[mo]}-{m.group(1).zfill(2)}"
            return m.group(0)
        text = self.DAY_MONTH_YEAR.sub(repl_d_m_y, text)

        def repl_m_d_y(m):
            mo = m.group(1).lower()
            if mo in self.MONTH_MAP:
                return f"{m.group(3)}-{self.MONTH_MAP[mo]}-{m.group(2).zfill(2)}"
            return m.group(0)
        text = self.MONTH_DAY_YEAR.sub(repl_m_d_y, text)

        def repl_m_y(m):
            mo = m.group(1).lower()
            if mo in self.MONTH_MAP:
                return f"{m.group(2)}:{self.MONTH_MAP[mo]}"
            return m.group(0)
        text = self.MONTH_YEAR.sub(repl_m_y, text)

        def repl_y_m(m):
            mo = m.group(2).lower()
            if mo in self.MONTH_MAP:
                return f"{m.group(1)}:{self.MONTH_MAP[mo]}"
            return m.group(0)
        text = self.YEAR_MONTH.sub(repl_y_m, text)
        return text

    def normalize_text_pipeline(self, text: str) -> str:
        """Executes the foundational structural regex cleanup processing pipeline."""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        phones = re.findall(self.PHONE_NUMBER, text)
        text = self.PHONE_NUMBER.sub(self.PHONE_PLACEHOLDER, text)
        text = self.STOCK_TICKER.sub(lambda m: m.group(0).replace(" ", ""), text)

        for reg, repl in self.EARLY_CLEANUP:
            text = reg.sub(repl, text)

        for word_regex, num in self.WRITTEN_NUMS.items():
            text = word_regex.sub(num, text)

        text = self.normalize_dates_smart(text)
        text = re.sub(r"(?<=\d),(?=\d)", "", text)

        for reg, repl in (self.FINANCIAL_CLEANUP + self.LATE_CLEANUP):
            text = reg.sub(repl, text)

        for p in phones:
            text = text.replace(self.PHONE_PLACEHOLDER, p.replace(" ", ""), 1)

        return text

    def process(self, text: str, strategy: str = "full") -> str:
        """
        Public structural entry API for evaluating preprocessors based on named strategies.

        :param text: Raw textual input context.
        :param strategy: Mode flag string (options: 'full', 'masked', 'standard_optimized', 'standard')
        """
        selected_strategy = self._strategies.get(strategy.lower())
        if not selected_strategy:
            raise ValueError(f"Strategy '{strategy}' is not registered. Choose from {list(self._strategies.keys())}")
        return selected_strategy.preprocess(text)


# ==========================================
# CONCRETE STRATEGY IMPLEMENTATIONS
# ==========================================

class FullPreprocessingStrategy(BasePreprocessorStrategy):
    """Deep structural preprocessor cleaning text pipelines, retaining currency/dates and lemmas."""
    def __init__(self, context: Preprocessor):
        self.ctx = context

    def preprocess(self, text: str) -> str:
        text = self.ctx.normalize_text_pipeline(text)
        tokens = text.split()
        tagged_tokens = pos_tag(tokens)

        lemmatized_tokens = []
        for word, tag in tagged_tokens:
            word_pos = self.ctx.get_wordnet_pos(tag)
            lemma = self.ctx.lemmatizer.lemmatize(word, pos=word_pos)
            if lemma not in self.ctx.custom_stopwords:
                lemmatized_tokens.append(lemma)
        return " ".join(lemmatized_tokens).strip()


class MaskedPreprocessingStrategy(BasePreprocessorStrategy):
    """Transforms numerical, date, and spatial targets into generic abstract labels."""
    def __init__(self, context: Preprocessor):
        self.ctx = context

    def preprocess(self, text: str) -> str:
        text = self.ctx.normalize_text_pipeline(text)
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
                word_pos = self.ctx.get_wordnet_pos(tag)
                lemma = self.ctx.lemmatizer.lemmatize(word, pos=word_pos)
                if lemma not in self.ctx.custom_stopwords:
                    processed_tokens.append(lemma)
        return re.sub(r'\s+', ' ', " ".join(processed_tokens)).strip()


class StandardOptimizedStrategy(BasePreprocessorStrategy):
    """Lightweight alphanumeric variant retaining underscores/word structures without regex pipelines."""
    def __init__(self, context: Preprocessor):
        self.ctx = context

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        tagged_tokens = pos_tag(tokens)

        processed_tokens = []
        for word, tag in tagged_tokens:
            word_pos = self.ctx.get_wordnet_pos(tag)
            lemma = self.ctx.lemmatizer.lemmatize(word, pos=word_pos)
            if lemma not in self.ctx.custom_stopwords:
                processed_tokens.append(lemma)
        return " ".join(processed_tokens).strip()


class StandardStrategy(BasePreprocessorStrategy):
    """Basic standard parser restricting sequences down purely to basic strings and spaces."""
    def __init__(self, context: Preprocessor):
        self.ctx = context

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        tokens = text.split()
        tagged_tokens = pos_tag(tokens)

        processed_tokens = []
        for word, tag in tagged_tokens:
            word_pos = self.ctx.get_wordnet_pos(tag)
            lemma = self.ctx.lemmatizer.lemmatize(word, pos=word_pos)
            if lemma not in self.ctx.custom_stopwords:
                processed_tokens.append(lemma)
        return " ".join(processed_tokens).strip()
