import re
import nltk

# ── Auto-download required NLTK data (safe to run multiple times) ─────────────
for _pkg in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4'):
    nltk.download(_pkg, quiet=True)
# ─────────────────────────────────────────────────────────────────────────────

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

QUESTION_WORDS = [
    "what", "why", "how", "when", "where", "who",
    "which", "whom", "define", "explain", "describe",
    "write", "list", "state", "mention", "give", "find",
    "calculate", "evaluate", "compare", "differentiate",
]

# ── Noise phrases removed from RAW TEXT before any sentence splitting ─────────
# These are document headers/footers that have no punctuation separator from
# the first real sentence, so sent_tokenize would merge them into one string
# and the whole string would be wrongly discarded.
NOISE_PHRASES = [
    "sample questions pdf",
    "sample questions docx",
    "sample questions",
    "english question paper",
    "english question",
    "question paper",
    "all the best",
    "time allowed",
    "maximum marks",
    "general instructions",
    "total marks",
    "section a",
    "section b",
    "answer all questions",
    "attempt any",
]

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def _strip_noise(text: str) -> str:
    """
    Remove document header/footer noise from raw text BEFORE sentence
    tokenisation.  This prevents noise strings from being merged with the
    first real question by sent_tokenize, which would cause that question
    to be silently dropped by the later question-detection filter.

    Strategy: for each noise phrase, replace it (case-insensitive) with a
    newline so the surrounding content still splits correctly.
    """
    for phrase in NOISE_PHRASES:
        # Replace the phrase and any trailing whitespace with a newline
        text = re.sub(
            re.escape(phrase) + r'[^\n.?!]*',   # phrase + any trailing non-punctuation
            '\n',
            text,
            flags=re.IGNORECASE
        )
    return text


def extract_questions(text: str) -> list:
    """
    Extract question sentences from raw text.
    A sentence qualifies as a question if it:
      - ends with '?', OR
      - starts with a recognised question keyword

    Pipeline:
      1. Strip document noise from raw text  ← KEY FIX
      2. Normalise whitespace
      3. Sentence-tokenise
      4. Filter by question markers
      5. Clean up numbering / stray whitespace
    """
    # Step 1 – remove noise BEFORE splitting into sentences
    text = _strip_noise(text)

    # Step 2 – normalise whitespace (preserve sentence boundaries)
    text = re.sub(r'[ \t]+', ' ', text)          # collapse spaces/tabs
    text = re.sub(r'\n\s*\n+', '\n', text)        # collapse blank lines
    text = re.sub(r'\n', '. ', text)               # turn line breaks into sentence breaks
    text = re.sub(r'\.\s*\.', '.', text)           # collapse double dots
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 3 – sentence tokenise
    sentences = sent_tokenize(text)

    questions = []

    for sentence in sentences:
        sentence = sentence.strip()

        # Remove leading numbering: "1.", "Q1.", "Q.1", "(1)", "1)"
        sentence = re.sub(r'^[\(\[]?\s*[Qq]?\.?\s*\d+[\)\]\.]\s*', '', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        # Minimum length guards
        if len(sentence) < 10:
            continue
        if len(sentence.split()) < 3:
            continue

        # Step 4 – question detection
        lower = sentence.lower()
        is_question = (
            sentence.endswith('?')
            or any(lower.startswith(word) for word in QUESTION_WORDS)
        )

        if is_question:
            questions.append(sentence)

    return questions


def clean_text(text: str) -> str:
    """
    Lowercase → remove punctuation → tokenise →
    remove stopwords → lemmatise → rejoin.
    Returns an empty string if nothing survives cleaning.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)

    tokens = word_tokenize(text)

    cleaned = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(cleaned)