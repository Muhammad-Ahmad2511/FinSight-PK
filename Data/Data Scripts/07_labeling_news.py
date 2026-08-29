import pandas as pd
import re

df = pd.read_csv('kse100_relevant_news.csv')

df['title_l'] = df['news'].fillna('').str.lower()
df['body_l'] = df['raw_text'].fillna('').str.lower()

UP_WORDS = ['rise', 'rises', 'rising', 'rose', 'increase', 'increases', 'increased', 'increasing',
            'surge', 'surges', 'surged', 'surging', 'soar', 'soars', 'soared', 'soaring',
            'jump', 'jumps', 'jumped', 'jumping', 'climb', 'climbs', 'climbed', 'climbing',
            'higher', 'highest', 'hike', 'hikes', 'hiked', 'hiking', 'grow', 'grows', 'grew', 'growing',
            'widen', 'widens', 'widened', 'widening', 'expand', 'expands', 'expanded',
            'accelerate', 'accelerated', 'accelerates', 'swell', 'swells', 'swelled',
            'balloon', 'ballooned', 'ballooning', 'up', 'gain', 'gains', 'gained', 'gaining']

DOWN_WORDS = ['fall', 'falls', 'falling', 'fell', 'decline', 'declines', 'declined', 'declining',
              'drop', 'drops', 'dropped', 'dropping', 'plunge', 'plunges', 'plunged', 'plunging',
              'crash', 'crashes', 'crashed', 'lower', 'lowest', 'cut', 'cuts', 'cutting',
              'reduce', 'reduces', 'reduced', 'reducing', 'shrink', 'shrinks', 'shrinking', 'shrunk',
              'slump', 'slumps', 'slumped', 'contract', 'contracts', 'contracted', 'narrow',
              'narrows', 'narrowed', 'ease', 'eases', 'eased', 'slow', 'slows', 'slowed', 'down',
              'dip', 'dips', 'dipped', 'tumble', 'tumbles', 'tumbled', 'weaken', 'weakens', 'weakened',
              'shortfall', 'slip', 'slips', 'slipped', 'edges down', 'edged down',
              'dive', 'dives', 'dived', 'diving', 'plummet', 'plummets', 'plummeted', 'plummeting',
              'batter', 'batters', 'battered', 'battering', 'hammer', 'hammers', 'hammered',
              'slam', 'slams', 'slammed']

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

POSITIVE_PATTERNS = [r'\bapprove[sd]?\b', r'\bapproval\b', r'agreement (?:reached|signed)',
                     r'\bsecures?\b', r'\bsecured\b', r'\brelief\b', r'\bresolved\b',
                     r'\bupgrad(?:e|es|ed)\b', r'stable outlook', r'positive outlook',
                     r'all-time high', r'record high', r'\brallies?\b', r'\bbullish\b',
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
                     r'\bmiss(?:es|ed)? target\b', r'bearish', r'\bworsen(?:s|ed|ing)?\b',
                     r'\bfails?\b', r'\bfailed\b', r'\bimpasse\b', r'\bstandoff\b',
                     r'\bhurts?\b', r'\bhurting\b', r'\bcrunch\b', r'\bturmoil\b',
                     r'\bvolatil(?:e|ity)\b', r'profit[- ]taking', r'sell-?off',
                     r'\bmeltdown\b', r'\bdistress\b', r'\bloses?\b.{0,20}points',
                     r'\bsheds?\b.{0,20}points', r'\braps\b', r'\bfail(?:s|ed)? to\b',
                     r'\bfailure to\b', r'\bbloodbath\b',
                     # FIX 2: added
                     r'\bbatter(?:s|ed|ing)?\b', r'\bhammer(?:s|ed|ing)?\b', r'\bslam(?:s|med)?\b',
                     r'\bplummet(?:s|ed|ing)?\b', r'\bdive[sd]?\b']

RATE_CUT = re.compile(r'(?:cuts?|lowers?|reduces?|slashe?s?) (?:the )?(?:policy |interest |discount )?rate|rate cut')
RATE_HIKE = re.compile(r'(?:hikes?|raises?|increases?) (?:the )?(?:policy |interest |discount )?rate|rate hike')
POS_RE = [re.compile(p) for p in POSITIVE_PATTERNS]
NEG_RE = [re.compile(p) for p in NEGATIVE_PATTERNS]

# sentence-scoped matching instead of raw character window.
# Split on sentence-ish boundaries; a metric and its direction word must
# fall in the SAME sentence/clause to be paired, not just within N chars
# of each other (which was pairing unrelated words like "tariff...surge"
# across two different clauses of one sentence).
SENT_SPLIT = re.compile(r'[.;]|(?<=[a-z0-9])\s(?=[A-Z][a-z])')  # also split on cap-letter clause starts

def get_sentences(text):
    """Return list of (start, end) spans for rough sentence/clause units."""
    spans = []
    last = 0
    for m in re.finditer(r'[.;,]', text):
        spans.append((last, m.end()))
        last = m.end()
    spans.append((last, len(text)))
    return spans


def find_direction(text, pos_start, pos_end, words, sentences):
    # find enclosing sentence/clause
    for (s, e) in sentences:
        if s <= pos_start < e:
            window = text[s:e]
            for w in words:
                if re.search(rf'\b{re.escape(w)}\b', window):
                    return True
            return False
    return False


def metric_scan(text, score, reasons):
    sentences = get_sentences(text)
    for metric in UP_GOOD_METRICS:
        pat = rf'\b{re.escape(metric)}\b(?!-taking)' if metric == 'profit' else rf'\b{re.escape(metric)}\b'
        for m in re.finditer(pat, text):
            if find_direction(text, m.start(), m.end(), UP_WORDS, sentences):
                score += 1; reasons.append(f'{metric} up (good)')
            elif find_direction(text, m.start(), m.end(), DOWN_WORDS, sentences):
                score -= 1; reasons.append(f'{metric} down (bad)')

    for metric in UP_BAD_METRICS:
        pat = rf'\b{re.escape(metric)}\b(?!-free)' if metric in ('tax', 'taxes') else rf'\b{re.escape(metric)}\b'
        for m in re.finditer(pat, text):
            if find_direction(text, m.start(), m.end(), UP_WORDS, sentences):
                score -= 1; reasons.append(f'{metric} up (bad)')
            elif find_direction(text, m.start(), m.end(), DOWN_WORDS, sentences):
                score += 1; reasons.append(f'{metric} down (good)')
    return score, reasons


def currency_scan(text, score, reasons):
    if re.search(r'\brupee\b.{0,20}\b(?:depreciat\w*|weaken\w*|falls?|falling|fell|slides?|slumps?|tumbles?|dive[sd]?|batter(?:s|ed)?|loses?\b.{0,10}(?:value|ground))\b', text) or \
       re.search(r'\b(?:depreciat\w*|weaken\w*)\b.{0,20}\brupee\b', text):
        score -= 1; reasons.append('rupee depreciates (bad)')
    if re.search(r'\brupee\b.{0,20}\b(?:appreciat\w*|strengthen\w*|gains?|rises?|rising|recovers?)\b', text) or \
       re.search(r'\b(?:appreciat\w*|strengthen\w*)\b.{0,20}\brupee\b', text):
        score += 1; reasons.append('rupee appreciates (good)')
    return score, reasons


def market_scan(text, score, reasons):
    if RATE_CUT.search(text):
        score += 2; reasons.append('rate cut (good)')
    if RATE_HIKE.search(text):
        score -= 2; reasons.append('rate hike (bad)')

    rally_broken = re.search(r'(?:snap|snaps|snapped|end|ends|ended|halt|halts|halted|broke|breaks|broken)\b.{0,20}\b(?:rally|rallies|bull run)\b', text)
    if re.search(r'\bbulls?\b|\brally\b|\brallies\b|record high|all-time high|\bsurges?\b.{0,15}points|\bgains?\b.{0,15}points|\bclimbs?\b.{0,15}points', text) and not rally_broken:
        score += 1; reasons.append('market up language')
    if re.search(r'\bbears?\b|\bplunges?\b|\bcrash(?:es|ed)?\b|\btumbles?\b|\bbloodbath\b|\bslides?\b|\bslid\b|\bslumps?\b|'
                 r'\bloses?\b.{0,20}points|\bsheds?\b.{0,20}points|\bslides?\b.{0,20}points|falls?\b.{0,20}points|drops?\b.{0,20}points|'
                 r'\bbatter(?:s|ed|ing)?\b|\bhammer(?:s|ed|ing)?\b|\bslam(?:s|med)?\b|\bplummet(?:s|ed|ing)?\b|\bdive[sd]?\b', text) or rally_broken:
        score -= 1; reasons.append('market down language')
    return score, reasons


def word_scan(text, score, reasons):
    pos_hits = sum(1 for p in POS_RE if p.search(text))
    neg_hits = sum(1 for p in NEG_RE if p.search(text))
    score += min(pos_hits, 2)
    score -= min(neg_hits, 2)
    if pos_hits: reasons.append(f'{pos_hits} positive word type(s)')
    if neg_hits: reasons.append(f'{neg_hits} negative word type(s)')
    return score, reasons


def score_row(title, body):
    score, reasons = 0, []
    score, reasons = metric_scan(title, score, reasons)
    score, reasons = market_scan(title, score, reasons)
    score, reasons = currency_scan(title, score, reasons)
    score, reasons = word_scan(title, score, reasons)

    if score == 0 and not reasons:
        score, reasons = metric_scan(body, score, reasons)
        score, reasons = market_scan(body, score, reasons)
        score, reasons = currency_scan(body, score, reasons)

    return score, '; '.join(reasons)


results = [score_row(t, b) for t, b in zip(df['title_l'], df['body_l'])]
df['sentiment_score'] = [r[0] for r in results]
df['reasons'] = [r[1] for r in results]


# distinguish genuine "no signal detected" from "signals cancelled out to zero"
def label(score, reasons):
    if score > 0:
        return 'Positive'
    if score < 0:
        return 'Negative'
    return 'No Signal' if not reasons else 'Neutral'


df['sentiment'] = [label(s, r) for s, r in zip(df['sentiment_score'], df['reasons'])]
print(df['sentiment'].value_counts())
print()


out = df[['date', 'news', 'what_it_affects', 'sentiment', 'source']].rename(
    columns={'news': 'News', 'what_it_affects': 'What It Affects',
             'sentiment': 'Sentiment', 'date': 'Date', 'source': 'Source'}
)
out.to_csv('kse100_relevant_news_with_sentiment.csv', index=False)