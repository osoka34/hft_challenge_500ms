---
name: hft-arxiv-features
description: >-
  Find one candidate feature in an arXiv, SSRN, Semantic Scholar, or OpenAlex
  paper and state it as a column this 500 ms tape can compute. Use when the
  user asks to search arXiv, SSRN, the archive, архив, or papers for a new
  feature.
---

# Paper feature search

Turn one paper into one column hypothesis. Wiring that column into code goes to `hft-feature-implement`. Whether it predicts `y` goes to `hft-feature-screen` only after implement. This skill stops at the hypothesis.

What is already logged lives in `reports/HFT_PAPERS.md` (status) and the verdict table in `reports/HFT_PAPER_EXPERIMENTS.md`. Quote those files; do not restate the paper list. New ledger lines follow `reports/HFT_LEDGER_ROW.md`.

## Steps

1. Read both reports.
   Done when the idea is marked todo, partial, done, blocked, or NOT OK there, or you can say it is absent.

2. If the user named a paper, open its abstract (any source below). If they did not, search in this order until one hit defines a **feature**, not a network architecture.
   Put the missing input in the query (depth buckets, trade flow, a leader, spread state, queue imbalance, OFI).
   Prefer a hit whose nearest logged column is absent or `todo`, not `done` / `NOT OK`.
   Take **title and authors from the live abstract** (Atom / API), not from a nickname in `HFT_PAPERS.md`. If the log's author line disagrees, correct the log when you write the ledger.
   Done when you have a title, authors, a stable id (arXiv / SSRN / DOI), and the sentence that defines the feature.

### Sources (try in order)

**1. arXiv API** — default for q-fin microstructure.
`https://export.arxiv.org/api/query?search_query=<q>+AND+(cat:q-fin.TR+OR+cat:q-fin.ST)&start=0&max_results=10&sortBy=relevance`
Atom: `title`, `id`, `summary`.

**2. Semantic Scholar** — when arXiv is thin, or to follow citations of a seed paper.
`https://api.semanticscholar.org/graph/v1/paper/search?query=<q>&limit=10&fields=title,abstract,externalIds,openAccessPdf,year,citationCount`
Seed citation / reference URLs (swap `citations` ↔ `references`):
- Cont OFI — `…/graph/v1/paper/ARXIV:1011.6402/citations?fields=title,abstract,externalIds&limit=20`
- Gould queue imb — `ARXIV:1512.03492`
- Albers BTC — `ARXIV:2108.09750`
- Lucchese volume LOB — `ARXIV:2211.13777`
- Sirignano–Cont universal — `ARXIV:1803.06917`
- Kolm deep OF — SSRN via search, abstract id `3900141` (no arXiv seed)
Prefer hits with `externalIds.ArXiv` or `openAccessPdf.url`.

**3. OpenAlex** — broad catalog + DOI when S2 rate-limits.
`https://api.openalex.org/works?search=<q>&per_page=10&filter=type:article|preprint`
Use `display_name`, `doi`, `abstract_inverted_index` (or landing page), `open_access.oa_url`.

**4. SSRN** — finance working papers often absent from arXiv (Kolm-style OF, Stoikov micro-price mirrors).
Search: `https://papers.ssrn.com/sol3/results.cfm?txtKey_Words=<q>` or look up by title. Id form: SSRN abstract id.

**5. RePEc / IDEAS** — econ working-paper trail when SSRN title search fails.
`https://ideas.repec.org/cgi-bin/htsearch?q=<q>` → abstract → PDF or DOI.

Skip as a first query: Google Scholar (no stable API, paywall noise), NBER (macro-heavy), Papers with Code (LOB CNN bias). Use them only to resolve a title the user already named.

3. Open the pdf only to copy that feature's formula. Map each input onto a stream we have.
   - `book`: L1 `bid_px`, `ask_px`, `bid_qty`, `ask_qty`, and volume buckets `bid_v01`..`bid_v11`, `ask_v01`..`ask_v11` on the 500 ms grid.
   - `bbo`: L1 quote changes.
   - `trades`: sparse 100 ms bars with `buy_qty` and `sell_qty`.
   - a leader instrument, joined as-of.
   - `targets`: legal `y` only, not an input.
   A formula that needs the message stream or a price at each book level stays a note that names the missing field. Sweep-style prices are in that class: the depth buckets are volumes, not a price ladder.
   Done when every input is one of our columns or an explicit missing field.

4. Hand the column to `hft-feature-implement`. Under a hunt, append a `todo` ledger row (`reports/HFT_LEDGER_ROW.md`) when the idea was absent. Outside a hunt, append only when the user asked to record it.
   Done when the answer has the paper id, the nearest logged column, and the hypothesis.

## Answer shape

```
# <column>

Paper: <title> (<authors>), <arXiv:id | SSRN:id | DOI>
Formula: <one line>
Inputs: <our columns> | missing: <field>
Nearest: `<column in HFT_PAPERS or the experiment log>` | absent
Logged: <todo | partial | done | NOT OK | blocked | absent>

Hypothesis for hft-feature-implement: <family>, horizon <y_500ms|y_2s|y_8s>
```
