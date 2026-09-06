"""
FinSight PK — Sentiment classifier v3: dependency-parsing based domain rules
+ FinBERT ensemble.

WHY v3 (vs the regex/char-distance v2):
  v2's find_nearest_direction() picked whichever UP/DOWN word was
  CHARACTER-CLOSEST to a metric mention within the same clause. That fixed
  the "checked UP list before DOWN list" bug, but character distance is
  still the wrong signal -- word order in English doesn't track "which verb
  governs which noun". Two verbs sharing one clause (via advcl/mark, e.g.
  "X jumps ... as Y falls") will have their direction words at whatever
  distance the sentence happens to put them, regardless of which noun each
  one is actually the subject of.

  v3 replaces character-distance with dependency-tree attribution: for each
  metric mention, walk UP the syntactic head chain (spaCy dependency parse)
  until it hits a token whose lemma is a known direction word. This finds
  the verb/noun that GRAMMATICALLY GOVERNS the metric, which is what
  actually determines "is this metric going up or down" -- not proximity.
  This is the standard shallow-SRL approach for exactly this
  "attach direction word to correct subject" problem in domain sentiment
  work (rule-based financial/economic NLP).

  Concretely this fixes, without any special-casing:
    "Current account deficit jumps ... as exports fall"
       -> deficit's head chain reaches "jumps" (deficit is nsubj of jumps)
       -> exports' head chain reaches "fall" (exports is nsubj of fall)
       (v2 grabbed whichever word was textually nearest to each metric,
       which for "deficit" was fine here but is NOT guaranteed in general --
       e.g. it fails as soon as the closer verb belongs to the OTHER clause)
    "Cut in policy rate underscores SBP's confidence ... gaining traction"
       -> rate's head chain: rate -[pobj]-> in -[prep]-> Cut (ROOT)
       -> "Cut" is a direction word (down) -- found via grammatical
          attachment (rate IS the object of "Cut in ..."), not by scanning
          for the nearest direction-shaped word in the sentence (which was
          "gaining", nowhere near "rate" grammatically)

  This is a drop-in replacement for the domain-rule layer only. market_scan
  (rate cut/hike phrasing, bull/bear language) and word_scan (generic
  positive/negative keyword layer) are unchanged from v2 -- those already
  operate on whole-clause pattern presence, not distance, so they were
  never affected by this class of bug.

REQUIREMENTS:
  pip install spacy
  python -m spacy download en_core_web_sm

CLI FLAGS:
  --input               raw input CSV (needs 'news' + 'raw_text' columns)
  --output              output CSV path
  --existing-finbert     path to a PREVIOUS merged output CSV to reuse
                         finbert_label/finbert_confidence from (by row
                         position), so you don't re-run FinBERT on GPU
  --spacy-model         spaCy model to use (default en_core_web_sm)
  --batch-size          spaCy nlp.pipe batch size (default 64)

Run in Colab with GPU enabled for the FinBERT pass (unless
--existing-finbert is used):
  !pip install -q transformers torch spacy
  !python -m spacy download en_core_web_sm
"""

import argparse
import re

import pandas as pd
import spacy

# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(description="FinSight PK sentiment classifier v3 (dependency-based)")
parser.add_argument("--input", default="kse100_relevant_news.csv",
                     help="Raw input CSV with 'news' and 'raw_text' columns")
parser.add_argument("--output", default="kse100_relevant_news_with_sentiment_v3.csv",
                     help="Output CSV path")
parser.add_argument("--existing-finbert", default=None,
                     help="Path to a previous merged CSV to reuse finbert_label/"
                          "finbert_confidence from (skips re-running FinBERT)")
parser.add_argument("--spacy-model", default="en_core_web_sm")
parser.add_argument("--batch-size", type=int, default=64)
args, _unknown_args = parser.parse_known_args()

print(f"Loading spaCy model '{args.spacy_model}'...")
nlp = spacy.load(args.spacy_model, disable=["ner"])

# ============================================================
# Domain vocabularies (unchanged from v2)
# ============================================================

UP_WORDS = {'rise', 'rises', 'rising', 'rose', 'risen', 'increase', 'increases', 'increased', 'increasing',
            'surge', 'surges', 'surged', 'surging', 'soar', 'soars', 'soared', 'soaring',
            'jump', 'jumps', 'jumped', 'jumping', 'climb', 'climbs', 'climbed', 'climbing',
            'higher', 'highest', 'high', 'highs', 'hike', 'hikes', 'hiked', 'hiking',
            'grow', 'grows', 'grew', 'growing', 'grown', 'widen', 'widens', 'widened', 'widening',
            'expand', 'expands', 'expanded', 'accelerate', 'accelerated', 'accelerates',
            'swell', 'swells', 'swelled', 'balloon', 'ballooned', 'ballooning',
            'up', 'gain', 'gains', 'gained', 'gaining'}

DOWN_WORDS = {'fall', 'falls', 'falling', 'fell', 'fallen', 'decline', 'declines', 'declined', 'declining',
              'drop', 'drops', 'dropped', 'dropping', 'plunge', 'plunges', 'plunged', 'plunging',
              'crash', 'crashes', 'crashed', 'lower', 'lowest', 'low', 'lows',
              'cut', 'cuts', 'cutting', 'reduce', 'reduces', 'reduced', 'reducing',
              'shrink', 'shrinks', 'shrinking', 'shrunk', 'slump', 'slumps', 'slumped',
              'contract', 'contracts', 'contracted', 'narrow', 'narrows', 'narrowed',
              'ease', 'eases', 'eased', 'slow', 'slows', 'slowed', 'down',
              'dip', 'dips', 'dipped', 'tumble', 'tumbles', 'tumbled',
              'weaken', 'weakens', 'weakened', 'slip', 'slips', 'slipped',
              'dive', 'dives', 'dived', 'diving', 'plummet', 'plummets', 'plummeted', 'plummeting',
              'batter', 'batters', 'battered', 'hammer', 'hammers', 'hammered', 'slam', 'slams', 'slammed'}

UNCHANGED_WORDS = {'unchanged', 'steady', 'held', 'holds', 'maintain', 'maintains', 'maintained',
                    'flat', 'stable'}

DEP_CURRENCY_WORDS = {'depreciate', 'depreciates', 'depreciated', 'depreciation',
                       'weaken', 'weakens', 'weakened', 'fall', 'falls', 'falling', 'fell',
                       'slide', 'slides', 'slid', 'slump', 'slumps', 'slumped',
                       'tumble', 'tumbles', 'tumbled', 'dive', 'dives', 'dived',
                       'batter', 'battered'}
APP_CURRENCY_WORDS = {'appreciate', 'appreciates', 'appreciated', 'appreciation',
                       'strengthen', 'strengthens', 'strengthened',
                       'gain', 'gains', 'gained', 'rise', 'rises', 'rising',
                       'recover', 'recovers', 'recovered'}

UP_GOOD_METRICS = ['export', 'exports', 'remittance', 'remittances', 'revenue', 'revenues',
                   'collection', 'collections', 'profit', 'profits', 'earning', 'earnings',
                   'reserve', 'reserves', 'foreign investment', 'fdi', 'fpi', 'gdp', 'growth',
                   'production', 'output', 'sales', 'deposit', 'deposits', 'dividend', 'dividends',
                   'inflow', 'inflows', 'market cap', 'points', 'shares', 'stocks']

UP_BAD_METRICS = ['inflation', 'cpi', 'consumer price index', 'price index', 'trade deficit',
                  'fiscal deficit', 'current account deficit',
                  'budget deficit', 'circular debt', 'public debt', 'external debt', 'sovereign debt',
                  'debt', 'import bill', 'tariff', 'tariffs', 'petrol price', 'fuel price',
                  'electricity price', 'gas price', 'power price', 'energy price', 'energy cost',
                  'energy costs', 'tax', 'taxes', 'taxation', 'policy rate', 'interest rate',
                  'discount rate', 'unemployment', 'loadshedding', 'load-shedding', 'losses',
                  'non-performing loans', 'npls', 'arrears']

ALL_METRIC_TERMS = UP_GOOD_METRICS + UP_BAD_METRICS

POSITIVE_PATTERNS = [r'\bapprove[sd]?\b', r'\bapproval\b', r'agreement (?:reached|signed)',
                     r'\bsecures?\b', r'\bsecured\b', r'\brelief\b', r'\bresolved\b',
                     r'\bupgrad(?:e|es|ed)\b', r'stable outlook', r'positive outlook',
                     r'all-time high', r'record high', r'\brallies?\b',
                     r'investor confidence', r'\bboost(?:s|ed)?\b', r'\bbailout approved\b',
                     r'loan approved', r'\bdisbursed\b', r'breakthrough',
                     r'(?:market|stocks?|shares?|economy|psx|kse)\b.{0,15}\brecovers?\b|\brecovery rally\b',
                     r'green light', r'\bendorse[sd]?\b', r'\bmilestone\b',
                     r'best[- ]perform', r'new high']

NEGATIVE_PATTERNS = [r'\breject(?:s|ed)?\b', r'\bdelay(?:s|ed)?\b', r'\bwarn(?:s|ed)?\b',
                     r'\bcrisis\b', r'\bcollapse[sd]?\b', r'\bunrest\b', r'\bprotest(?:s|ed)?\b',
                     r'\bstrike[sd]?\b', r'\bdowngrad(?:e|es|ed)\b', r'\bdefault\b',
                     r'instability', r'\bunstable\b', r'\bshortfall\b', r'\bscandal\b',
                     r'\bfraud\b', r'\bcorruption\b', r'\bban(?:ned|s)?\b', r'\bhalt(?:s|ed)?\b',
                     r'\bsuspend(?:s|ed)?\b', r'\bthreat(?:en|ens|ened)?\b', r'\boutflows?\b',
                     r'\bpenalty\b', r'\bfined?\b', r'\bslowdown\b', r'\bstruggl(?:e|es|ed|ing)\b',
                     r'\bmiss(?:es|ed)? target\b', r'\bworsen(?:s|ed|ing)?\b',
                     r'\bfails?\b', r'\bfailed\b', r'\bimpasse\b', r'\bstandoff\b',
                     r'\bhurts?\b', r'\bhurting\b', r'\bcrunch\b', r'\bturmoil\b',
                     r'\bvolatil(?:e|ity)\b', r'profit[- ]taking', r'sell-?off',
                     r'\bmeltdown\b', r'\bdistress\b', r'\bloses?\b.{0,20}points',
                     r'\bsheds?\b.{0,20}points', r'\braps\b', r'\bfail(?:s|ed)? to\b',
                     r'\bfailure to\b', r'\bbloodbath\b',
                     r'\bbatter(?:s|ed|ing)?\b', r'\bhammer(?:s|ed|ing)?\b', r'\bslam(?:s|med)?\b',
                     r'\bplummet(?:s|ed|ing)?\b', r'\bdive[sd]?\b']

RATE_CUT = re.compile(r'(?:cuts?|lowers?|reduces?|slashe?s?) (?:the )?(?:policy |interest |discount )?rate|rate cut')
RATE_HIKE = re.compile(r'(?:hikes?|raises?|increases?) (?:the )?(?:policy |interest |discount )?rate|rate hike')

NEGATED_BAD_OUTCOME = re.compile(
    r'\b(?:does not|doesn\'t|did not|didn\'t|won\'t|will not|unlikely to|'
    r'not expected to|no signs? of|no risk of|not likely to)\b.{0,60}?'
    r'\b(?:trigger|cause|spark|lead to|result in)\b.{0,30}?'
    r'\b(?:crisis|collapse|recession|meltdown|crash|default|turmoil)\b'
)

NEGATION_CUES = re.compile(
    r'\b(?:not|no|never|doesn\'t|does not|didn\'t|did not|won\'t|will not|'
    r'isn\'t|is not|wasn\'t|was not|cannot|can\'t|unable to|unlikely to|'
    r'fails? to|failed to|avoid(?:s|ed)?)\b'
)

POS_RE = [re.compile(p) for p in POSITIVE_PATTERNS]
NEG_RE = [re.compile(p) for p in NEGATIVE_PATTERNS]


# ============================================================
# Dependency-based direction attribution (the v3 core change)
# ============================================================

def _direction_for(lemma, text_lower, up_set, down_set, unchanged_set=None):
    if unchanged_set and (lemma in unchanged_set or text_lower in unchanged_set):
        return 'unchanged'
    if lemma in up_set or text_lower in up_set:
        return 'up'
    if lemma in down_set or text_lower in down_set:
        return 'down'
    return None


def _is_negated(tok):
    return any(child.dep_ == 'neg' for child in tok.children)


def governing_direction(metric_token, up_set, down_set, unchanged_set=None, max_hops=6):
    """Walk up the dependency tree from metric_token, within its own
    sentence, until a direction word is found. Returns (direction, negated)
    or (None, False) if the sentence never attaches a direction word to
    this metric at all -- correctly yielding 'no signal' rather than
    guessing from an unrelated nearby word."""
    sent = metric_token.sent
    cur = metric_token
    for _ in range(max_hops):
        if cur.dep_ == 'ROOT':
            break
        nxt = cur.head
        if nxt.sent != sent:
            break
        d = _direction_for(nxt.lemma_.lower(), nxt.text.lower(), up_set, down_set, unchanged_set)
        if d is not None:
            return d, _is_negated(nxt)
        if nxt.dep_ == 'ROOT' or nxt == cur:
            break
        cur = nxt
    return None, False


def find_metric_heads(doc, phrase):
    """Locate occurrences of a (possibly multi-word) metric phrase and
    return the syntactic root token of each match."""
    words = phrase.split()
    n = len(words)
    heads = []
    toks = [t.text.lower() for t in doc]
    for i in range(len(toks) - n + 1):
        if toks[i:i + n] == words:
            heads.append(doc[i:i + n].root)
    return heads


def metric_scan_dep(doc, score, reasons):
    for metric in UP_GOOD_METRICS:
        for head in find_metric_heads(doc, metric):
            direction, negated = governing_direction(head, UP_WORDS, DOWN_WORDS, UNCHANGED_WORDS)
            if direction is None or direction == 'unchanged':
                if direction == 'unchanged':
                    reasons.append(f'{metric} unchanged (no signal)')
                continue
            if negated:
                direction = 'down' if direction == 'up' else 'up'
            if direction == 'up':
                score += 1; reasons.append(f'{metric} up (good)')
            else:
                score -= 1; reasons.append(f'{metric} down (bad)')
    for metric in UP_BAD_METRICS:
        for head in find_metric_heads(doc, metric):
            direction, negated = governing_direction(head, UP_WORDS, DOWN_WORDS, UNCHANGED_WORDS)
            if direction is None or direction == 'unchanged':
                if direction == 'unchanged':
                    reasons.append(f'{metric} unchanged (no signal)')
                continue
            if negated:
                direction = 'down' if direction == 'up' else 'up'
            if direction == 'up':
                score -= 1; reasons.append(f'{metric} up (bad)')
            else:
                score += 1; reasons.append(f'{metric} down (good)')
    return score, reasons


def currency_scan_dep(doc, score, reasons):
    for head in find_metric_heads(doc, 'rupee'):
        direction, negated = governing_direction(head, APP_CURRENCY_WORDS, DEP_CURRENCY_WORDS)
        if direction is None:
            continue
        if negated:
            direction = 'down' if direction == 'up' else 'up'
        if direction == 'up':
            score += 1; reasons.append('rupee appreciates (good)')
        else:
            score -= 1; reasons.append('rupee depreciates (bad)')
    return score, reasons


def title_has_domain_topic(text_lower):
    for metric in ALL_METRIC_TERMS:
        if re.search(rf'\b{re.escape(metric)}\b', text_lower):
            return True
    if re.search(r'\brupee\b', text_lower):
        return True
    if RATE_CUT.search(text_lower) or RATE_HIKE.search(text_lower):
        return True
    if re.search(r'\bbulls?\b|\bbullish\b|\bbears?\b|\bbearish\b|\brally\b|\brallies\b', text_lower):
        return True
    return False


def market_scan(text, score, reasons):
    if RATE_CUT.search(text):
        score += 2; reasons.append('rate cut (good)')
    if RATE_HIKE.search(text):
        score -= 2; reasons.append('rate hike (bad)')
    if NEGATED_BAD_OUTCOME.search(text):
        score += 2; reasons.append('negated bad outcome (good)')
    rally_broken = re.search(r'(?:snap|snaps|snapped|end|ends|ended|halt|halts|halted|broke|breaks|broken)\b.{0,20}\b(?:rally|rallies|bull run)\b', text)
    if re.search(r'\bbulls?\b|\bbullish\b|\brally\b|\brallies\b|record high|all-time high|\bsurges?\b.{0,15}points|\bgains?\b.{0,15}points|\bclimbs?\b.{0,15}points', text) and not rally_broken:
        score += 1; reasons.append('market up language')
    if re.search(r'\bbears?\b|\bbearish\b|\bplunges?\b|\bcrash(?:es|ed)?\b|\btumbles?\b|\bbloodbath\b|\bslides?\b|\bslid\b|\bslumps?\b|'
                 r'\bloses?\b.{0,20}points|\bsheds?\b.{0,20}points|\bslides?\b.{0,20}points|falls?\b.{0,20}points|drops?\b.{0,20}points|'
                 r'\bbatter(?:s|ed|ing)?\b|\bhammer(?:s|ed|ing)?\b|\bslam(?:s|med)?\b|\bplummet(?:s|ed|ing)?\b|\bdive[sd]?\b', text) or rally_broken:
        score -= 1; reasons.append('market down language')
    return score, reasons


def get_sentences(text):
    spans = []
    last = 0
    for m in re.finditer(r'[.;,]', text):
        spans.append((last, m.end()))
        last = m.end()
    spans.append((last, len(text)))
    return spans


def is_negated_regex(text, match_start, sentences, lookback_chars=40):
    for (s, e) in sentences:
        if s <= match_start < e:
            window_start = max(s, match_start - lookback_chars)
            window = text[window_start:match_start]
            return bool(NEGATION_CUES.search(window))
    return False


def word_scan(text, score, reasons):
    sentences = get_sentences(text)
    pos_hits = 0
    neg_hits = 0
    for p in POS_RE:
        m = p.search(text)
        if m and not is_negated_regex(text, m.start(), sentences):
            pos_hits += 1
    for p in NEG_RE:
        m = p.search(text)
        if m and not is_negated_regex(text, m.start(), sentences):
            neg_hits += 1
    score += min(pos_hits, 2)
    score -= min(neg_hits, 2)
    if pos_hits: reasons.append(f'{pos_hits} positive word type(s)')
    if neg_hits: reasons.append(f'{neg_hits} negative word type(s)')
    return score, reasons


def score_row(title, title_doc, body, body_doc):
    domain_score, domain_reasons = 0, []
    domain_score, domain_reasons = metric_scan_dep(title_doc, domain_score, domain_reasons)
    domain_score, domain_reasons = market_scan(title, domain_score, domain_reasons)
    domain_score, domain_reasons = currency_scan_dep(title_doc, domain_score, domain_reasons)

    generic_score, generic_reasons = word_scan(title, 0, [])

    title_topic_present = title_has_domain_topic(title) or bool(domain_reasons) or bool(generic_reasons)
    fallback_allowed = not title_topic_present

    if fallback_allowed and body_doc is not None:
        domain_score, domain_reasons = metric_scan_dep(body_doc, domain_score, domain_reasons)
        domain_score, domain_reasons = market_scan(body, domain_score, domain_reasons)
        domain_score, domain_reasons = currency_scan_dep(body_doc, domain_score, domain_reasons)
        generic_score, generic_reasons = word_scan(body, generic_score, generic_reasons)

    return domain_score, domain_reasons, generic_score, generic_reasons


def regex_label(domain_score, generic_score, has_reasons):
    total = domain_score + generic_score
    if total > 0:
        return 'Positive'
    if total < 0:
        return 'Negative'
    return 'No Signal' if not has_reasons else 'Neutral'


# ============================================================
# PART 1 — Dependency-based domain classifier
# ============================================================

df = pd.read_csv(args.input)
df['title_l'] = df['title'].fillna('').str.lower()
df['body_l'] = df['raw_text'].fillna('').str.lower()

print(f"Parsing {len(df)} titles with spaCy...")
title_docs = list(nlp.pipe(df['title_l'].tolist(), batch_size=args.batch_size))

# Only parse bodies for rows whose title raises no domain topic at all
# (mirrors v2's fallback gating, computed up front so we don't parse every
# body when most titles already carry a clear signal).
needs_body = [not title_has_domain_topic(t) for t in df['title_l']]
body_docs = [None] * len(df)
body_idx = [i for i, need in enumerate(needs_body) if need]
if body_idx:
    print(f"Parsing {len(body_idx)} article bodies (titles with no domain topic)...")
    parsed = nlp.pipe(df['body_l'].iloc[body_idx].tolist(), batch_size=args.batch_size)
    for i, d in zip(body_idx, parsed):
        body_docs[i] = d

print("Scoring rows...")
results = [
    score_row(t, td, b, bd)
    for t, td, b, bd in zip(df['title_l'], title_docs, df['body_l'], body_docs)
]
df['domain_score'] = [r[0] for r in results]
df['domain_reasons'] = ['; '.join(r[1]) for r in results]
df['generic_score'] = [r[2] for r in results]
df['generic_reasons'] = ['; '.join(r[3]) for r in results]
df['regex_score'] = df['domain_score'] + df['generic_score']
df['regex_reasons'] = df['domain_reasons'].str.cat(df['generic_reasons'], sep='; ').str.strip('; ')

df['regex_label'] = [
    regex_label(d, g, bool(dr) or bool(gr))
    for d, g, dr, gr in zip(df['domain_score'], df['generic_score'], df['domain_reasons'], df['generic_reasons'])
]
print(df['regex_label'].value_counts())

# ============================================================
# PART 2 — FinBERT (unchanged from v2)
# ============================================================

if args.existing_finbert:
    print(f"\nReusing FinBERT results from {args.existing_finbert} (skipping GPU pass)...")
    prev = pd.read_csv(args.existing_finbert)
    if len(prev) != len(df):
        raise ValueError(
            f"--existing-finbert row count ({len(prev)}) doesn't match "
            f"--input row count ({len(df)}) -- can't align by position."
        )
    df['finbert_label'] = prev['finbert_label'].values
    df['finbert_confidence'] = prev['finbert_confidence'].values
else:
    print("\nLoading FinBERT...")
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    print("Using GPU" if device == 0 else "Using CPU (this will be slower)")

    clf = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert", device=device)

    headlines = df['title'].fillna('').tolist()
    print(f"Running FinBERT on {len(headlines)} headlines (batched)...")
    BATCH_SIZE = 32
    finbert_labels, finbert_scores = [], []
    for i in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[i:i + BATCH_SIZE]
        outputs = clf(batch, truncation=True, max_length=128)
        for o in outputs:
            finbert_labels.append(o['label'].capitalize())
            finbert_scores.append(o['score'])
        if i % (BATCH_SIZE * 50) == 0:
            print(f"  {i}/{len(headlines)}")
    df['finbert_label'] = finbert_labels
    df['finbert_confidence'] = finbert_scores

print("\nFinBERT label distribution:")
print(df['finbert_label'].value_counts())

# ============================================================
# PART 3 — Ensemble (unchanged from v2)
# ============================================================

def combine_labels(row):
    """Ensemble priority (v3.1): domain rule > agreement > confidence-
    gated disagreement > finbert-only.

    On disagreement (domain rule silent, regex generic-word-scan and
    FinBERT disagree), only let FinBERT override the regex label if
    FinBERT is reasonably confident (>= DISAGREEMENT_CONFIDENCE_THRESHOLD).
    Blindly trusting FinBERT on every disagreement was flipping cases like
    "Stocks recover 349 points" (unambiguous positive) to Negative purely
    because FinBERT's own confidence on that headline was mediocre (~0.43)
    -- lower than the confidence backing plenty of headlines FinBERT gets
    right. A low-confidence FinBERT call is exactly the situation where a
    domain-relevant regex signal (even the weaker generic layer, not the
    metric-direction domain layer) should be allowed to win instead of
    being silently discarded."""
    domain_score = row['domain_score']
    regex_l = row['regex_label']
    fb_l = row['finbert_label']
    fb_conf = row['finbert_confidence']

    if domain_score != 0:
        domain_label = 'Positive' if domain_score > 0 else 'Negative'
        return domain_label, 'regex_domain', (domain_label == fb_l)

    if regex_l == 'No Signal':
        return fb_l, 'finbert_only', None

    if regex_l == fb_l:
        return regex_l, 'agreement', True

    if fb_conf is not None and fb_conf >= DISAGREEMENT_CONFIDENCE_THRESHOLD:
        return fb_l, 'disagreement_trust_finbert', False
    return regex_l, 'disagreement_trust_regex_low_finbert_conf', False


DISAGREEMENT_CONFIDENCE_THRESHOLD = 0.60


combined = df.apply(combine_labels, axis=1, result_type='expand')
df['final_sentiment'] = combined[0]
df['ensemble_method'] = combined[1]
df['regex_finbert_agree'] = combined[2]

print("\nFinal ensemble label distribution:")
print(df['final_sentiment'].value_counts())
print("\nEnsemble method breakdown:")
print(df['ensemble_method'].value_counts())

comparable = df[df['regex_finbert_agree'].notna()]
print(f"\nRows where both models had an opinion: {len(comparable)} / {len(df)}")
print(f"Agreement rate on those rows: {comparable['regex_finbert_agree'].mean():.1%}")

df['source'] = 'Dawn'
out = df[['date', 'title', 'url', 'final_sentiment', 'source',
          'regex_label', 'domain_score', 'generic_score', 'regex_reasons',
          'finbert_label', 'finbert_confidence', 'ensemble_method']].rename(columns={
    'title': 'News', 'url': 'URL',
    'final_sentiment': 'Sentiment', 'date': 'Date', 'source': 'Source'
})
out.to_csv(args.output, index=False)
print(f"\nSaved -> {args.output}")