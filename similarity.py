import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Tuneable parameters ───────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.60   # hybrid score must exceed this to be "repeated"
W_COSINE             = 0.55   # weight for TF-IDF cosine
W_JACCARD            = 0.45   # weight for Jaccard token overlap
MIN_TOPIC_WORDS      = 1      # shared topic words required
LENGTH_RATIO_LIMIT   = 2.5    # token-count ratio beyond which penalty kicks in

# Words that describe HOW a question is asked, not WHAT it is about.
# Two questions sharing ONLY these words are NOT considered similar.
QUESTION_VERBS = {
    "what", "why", "how", "when", "where", "who", "which", "whom",
    "define", "explain", "describe", "write", "list", "state",
    "mention", "give", "find", "calculate", "evaluate", "compare",
    "differentiate", "discuss", "identify", "name", "illustrate",
    "outline", "summarize", "analyse", "analyze", "derive", "prove",
    "show", "determine", "classify", "examine",
    "is", "are", "do", "does", "can", "has", "have",
}
# ─────────────────────────────────────────────────────────────────────────────


def _jaccard(tokens_a: list, tokens_b: list) -> float:
    """Jaccard similarity between two token lists (set-based)."""
    sa, sb = set(tokens_a), set(tokens_b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _topic_words(tokens: list) -> set:
    """
    Return content tokens that identify WHAT the question is about.
    Filters out question verbs and very short tokens.
    """
    return {t for t in tokens if t not in QUESTION_VERBS and len(t) >= 4}


def _length_penalty(len_a: int, len_b: int) -> float:
    """
    Returns a multiplier in (0, 1].
    Penalises pairs where one question is much longer than the other,
    because shared tokens represent a smaller fraction of meaning.
    Penalty only activates when ratio > LENGTH_RATIO_LIMIT.
    """
    if len_a == 0 or len_b == 0:
        return 0.0
    ratio = max(len_a, len_b) / min(len_a, len_b)
    if ratio <= LENGTH_RATIO_LIMIT:
        return 1.0
    # Linear decay toward 0 as ratio grows beyond the limit
    return max(0.0, 1.0 - (ratio - LENGTH_RATIO_LIMIT) / (10.0 - LENGTH_RATIO_LIMIT))


def classify_questions(
    cleaned_questions: list,
    original_questions: list,
) -> tuple:
    """
    Classify questions as repeated vs rare.

    Parameters
    ----------
    cleaned_questions  : preprocessed strings (from preprocess.clean_text)
    original_questions : raw strings (for CSV output and display)

    Returns
    -------
    (repeated_indexes, rare_indexes)  — sorted lists of int positions
    """
    n = len(cleaned_questions)

    if n < 2:
        return [], list(range(n))

    # Pre-tokenise for Jaccard and topic-word checks
    token_lists = [q.split() for q in cleaned_questions]

    # ── TF-IDF cosine matrix ──────────────────────────────────────────────────
    vectorizer   = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_questions)
    cos_matrix   = cosine_similarity(tfidf_matrix)

    repeated_indexes = set()
    rare_indexes     = set(range(n))
    similarity_report = []

    for i in range(n):
        for j in range(i + 1, n):
            toks_i = token_lists[i]
            toks_j = token_lists[j]

            # ── Guard: must share at least one topic word ─────────────────────
            topic_i = _topic_words(toks_i)
            topic_j = _topic_words(toks_j)
            shared_topics = topic_i & topic_j

            raw_cos = float(cos_matrix[i][j])
            jac     = _jaccard(toks_i, toks_j)

            if len(shared_topics) < MIN_TOPIC_WORDS:
                # No shared topic word → cannot be a repeat regardless of score
                similarity_report.append([
                    original_questions[i],
                    original_questions[j],
                    round(raw_cos, 4),
                    round(jac, 4),
                    0.0,
                    "rare (no topic overlap)",
                ])
                continue

            # ── Hybrid score with length penalty ──────────────────────────────
            penalty = _length_penalty(len(toks_i), len(toks_j))
            hybrid  = (W_COSINE * raw_cos + W_JACCARD * jac) * penalty

            decision = "rare"
            if hybrid >= SIMILARITY_THRESHOLD:
                repeated_indexes.add(i)
                repeated_indexes.add(j)
                rare_indexes.discard(i)
                rare_indexes.discard(j)
                decision = "repeated"

            similarity_report.append([
                original_questions[i],
                original_questions[j],
                round(raw_cos, 4),
                round(jac, 4),
                round(hybrid, 4),
                decision,
            ])

    repeated_list = sorted(repeated_indexes)
    rare_list     = sorted(rare_indexes)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    _save_csv(
        "output/repeated_questions.csv",
        [[original_questions[i]] for i in repeated_list],
        ["Repeated Questions"],
    )
    _save_csv(
        "output/rare_questions.csv",
        [[original_questions[i]] for i in rare_list],
        ["Rare Questions"],
    )
    _save_csv(
        "output/similarity_report.csv",
        similarity_report,
        ["Question 1", "Question 2", "TF-IDF Cosine", "Jaccard", "Hybrid Score", "Decision"],
    )

    return repeated_list, rare_list


def _save_csv(path: str, rows: list, columns: list) -> None:
    try:
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    except Exception as e:
        print(f"CSV save error ({path}): {e}")