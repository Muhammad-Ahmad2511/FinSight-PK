"""
Step 6 — Filter Dawn Business scrape down to news relevant to KSE-100.

Input:  dawn_business_2015_to_2026.csv  (columns: date, title, url, raw_text)
Output: kse100_relevant_news.csv        (columns: date, news, what_it_affects, source)

Approach: keyword/regex matching against ~20 macro/sector/market categories
that plausibly move the KSE-100, with section-based exclusion (Sport/Branded/
Prism dropped outright; World-section items dropped unless clearly Pakistan-
tied OR globally market-moving e.g. oil price / Fed decisions).
"""

import pandas as pd
import re

df = pd.read_csv('dawn_business_2015_to_2026.csv', encoding='utf-8-sig')

# Combine title + raw_text for matching (lowercased)
df['blob'] = (df['title'].fillna('') + ' ' + df['raw_text'].fillna('')).str.lower()

# Section tag is the first word of raw_text (Business, Pakistan, World, Sport, Branded, Prism...)
df['section'] = df['raw_text'].fillna('').str.extract(r'^(\w+)')

PAK_SPECIFIC = re.compile(
    r'\bpakistan\b|\bpsx\b|\bkse[- ]?100\b|\bsbp\b|\bfbr\b|\brupee\b|\bpkr\b|\bogra\b|'
    r'\bislamabad\b|\bkarachi\b|\blahore\b|\bpeshawar\b|\bquetta\b|\bnepra\b|\bshehbaz\b|'
    r'\bishaq dar\b|\baurangzeb\b'
)

# Category -> list of keyword patterns (regex, word-boundary where sensible)
CATEGORIES = {
    'Stock Market (Direct)': [
        r'\bkse[- ]?100\b', r'\bpsx\b', r'pakistan stock exchange', r'\bbourse\b',
        r'stock market', r'bulls\b', r'bears\b', r'index (?:surge|fall|drop|gain|close|hit)',
        r'\bbrokerage\b', r'market cap', r'all-time high', r'circuit breaker'
    ],
    'Monetary Policy / Interest Rates': [
        r'\bsbp\b', r'state bank of pakistan', r'policy rate', r'discount rate',
        r'interest rate', r'monetary policy', r'repo rate'
    ],
    'Inflation / CPI': [
        r'\bcpi\b', r'inflation', r'consumer price'
    ],
    'Currency / Exchange Rate': [
        r'\brupee\b', r'\bpkr\b', r'exchange rate', r'currency', r'\bdollar\b', r'devaluation'
    ],
    'IMF / External Financing': [
        r'\bimf\b', r'international monetary fund', r'bailout', r'loan tranche',
        r'staff-level agreement', r'extended fund facility', r'\beff\b',
        r'world bank', r'asian development bank', r'\badb\b', r'saudi (?:loan|deposit|fund)',
        r'\bgcc\b.*(?:loan|deposit)', r'debt restructuring'
    ],
    'Fiscal Policy / Budget / Taxes': [
        r'\bbudget\b', r'\bfbr\b', r'federal board of revenue', r'\btax(es|ation)?\b',
        r'fiscal deficit', r'fiscal policy', r'mini-budget', r'subsid(?:y|ies)',
        r'\bpsdp\b', r'revenue target', r'\bimport duty\b', r'customs duty',
        r'\bsoes?\b', r'state-owned enterprise', r'\btariff(s)?\b', r'\bduty\b', r'\bduties\b'
    ],
    'Current Account / Trade': [
        r'current account', r'trade deficit', r'trade balance', r'\bexports?\b',
        r'\bimports?\b', r'balance of payments'
    ],
    'Foreign Reserves': [
        r'foreign (?:exchange )?reserves', r'\bforex reserves\b'
    ],
    'Foreign Investment / FPI': [
        r'foreign (?:direct )?investment', r'\bfdi\b', r'foreign portfolio',
        r'foreign investors?', r'\bmsci\b'
    ],
    'Banking Sector': [
        r'\bbanking sector\b', r'\bbanks?\'? (?:profit|earnings|loans|deposits|advances)',
        r'\bnbp\b', r'\bhbl\b', r'\bubl\b', r'meezan bank', r'islamic banking',
        r'non-performing loans', r'\bnpls\b', r'\badr\b.*bank', r'central bank'
    ],
    'Oil & Gas / Energy Sector': [
        r'\bpetroleum price\b', r'\bpetrol\b', r'\bdiesel\b', r'\bogra\b',
        r'\bpetrol(?:eum)? levy\b', r'\bgas (?:price|tariff|sale|shortage)\b',
        r'\blng\b', r'circular debt', r'power sector', r'electricity (?:tariff|price)', r'energy sector',
        r'independent power produc', r'oil (?:price|import|marketing|exploration)', r'\bpso\b',
        r'\boil and gas\b', r'\bbrent\b', r'\bcrude\b', r'oil prices?', r'\bopec\b'
    ],
    'Cement Sector': [
        r'\bcement\b'
    ],
    'Fertilizer Sector': [
        r'\bfertiliser\b', r'\bfertilizer\b', r'\burea\b'
    ],
    'Textile Sector': [
        r'\btextile\b', r'\bapparel\b', r'\bgarments?\b'
    ],
    'Automobile Sector': [
        r'\bauto(?:mobile)?\s?(?:sector|industry|sales)\b', r'\bpama\b'
    ],
    'Credit Rating': [
        r'credit rating', r"\bmoody'?s\b", r'\bfitch\b', r"\bs&p\b", r'sovereign rating'
    ],
    'GDP / Economic Growth': [
        r'\bgdp\b', r'economic growth', r'growth target', r'economic survey'
    ],
    'Political Stability / Government': [
        r'\bgeneral elections?\b', r'no[- ]confidence', r'martial law', r'\bimran khan\b',
        r'\bshehbaz sharif\b', r'political crisis', r'government collapse', r'coalition government',
        r'political instability', r'\bpti\b', r'\bpml-n\b', r'\bppp\b', r'dissolv(?:e|ed|ing) assembl'
    ],
    'Privatization / Divestment': [
        r'privatisation', r'privatization', r'\bpia\b', r'divestment', r'stake sale'
    ],
    'Debt / External Debt': [
        r'external debt', r'public debt', r'sovereign bonds?', r'eurobond', r'\bsukuk\b'
    ],
    'Global Markets / Fed': [
        r'federal reserve', r'\bfed\b.{0,15}(?:rate|hike|cut)', r'wall street',
        r'dow jones', r'\bs&p 500\b', r'global stock market', r'\bnasdaq\b'
    ],
}

# Categories that matter for KSE-100 even without an explicit Pakistan mention —
# global oil moves and Fed decisions hit PSX energy/flow-sensitive stocks regardless
# of whether the headline names Pakistan directly.
GLOBAL_RELEVANT_CATEGORIES = {'Oil & Gas / Energy Sector', 'Global Markets / Fed'}


def match_categories(text):
    hits = []
    for cat, patterns in CATEGORIES.items():
        for pat in patterns:
            if re.search(pat, text):
                hits.append(cat)
                break
    return hits


df['categories'] = df['blob'].apply(match_categories)
df['num_hits'] = df['categories'].apply(len)
df['pak_specific'] = df['blob'].apply(lambda t: bool(PAK_SPECIFIC.search(t)))
df['has_global_relevant'] = df['categories'].apply(
    lambda cats: any(c in GLOBAL_RELEVANT_CATEGORIES for c in cats)
)

# Drop Sport/Branded/Prism sections outright; drop World-section items unless
# clearly Pakistan-tied or globally market-moving (oil/Fed)
DROP_SECTIONS = {'Sport', 'Branded', 'Prism'}
keep_mask = (df['num_hits'] > 0) & (~df['section'].isin(DROP_SECTIONS))
keep_mask &= ~((df['section'] == 'World') & (~df['pak_specific']) & (~df['has_global_relevant']))

filtered = df[keep_mask].copy()
filtered['what_it_affects'] = filtered['categories'].apply(lambda x: '; '.join(x))

out = filtered[['date', 'title', 'what_it_affects', 'url', 'raw_text']].rename(
    columns={'title': 'news', 'url': 'source'}
)
out = out.sort_values(['date', 'news']).reset_index(drop=True)

print('Total rows:', len(df))
print('Filtered rows:', len(out))
print(out['what_it_affects'].value_counts().head(20))

out.to_csv('kse100_relevant_news.csv', index=False)