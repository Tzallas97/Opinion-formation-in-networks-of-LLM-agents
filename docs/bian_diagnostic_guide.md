# Οδηγός: Bian 5-dim diagnostic (`tools/bian_diagnostic.py`)

Standalone εργαλείο που «μετράει τις προδιαθέσεις» ενός LLM πριν το χρησιμοποιήσεις
σε opinion-dynamics run. Πέντε σύντομες probes, γραμμένες ώστε να τρέχουν μία φορά
ανά μοντέλο και να καταγράφονται στο `run_metrics.json` ως validity self-doc.

Πηγή: **Bian, N. et al. (2025). Social Simulations with Large Language Model Risk
Utopian Illusion. arXiv:2510.21180** (`papers/Social Simulations with Large Language
Model Risk Utopian Illusion - 2510.21180v1.pdf`). Το paper δείχνει ότι LLM social
simulations υπόκεινται σε 3 συστηματικά biases με πηγή τα pretraining N-grams
(occupation over-representation) και τα preference datasets (positivity). Το P28
methodology paper κάνει το επιχείρημα: **κάθε opinion-dynamics paper οφείλει να
αναφέρει αυτά τα scores για το μοντέλο του.**

## Τι μετράει κάθε probe (υψηλότερο = πιο biased σε κάθε άξονα)

| Score | Τι μετράει | Πώς | Τυπικό εύρος* |
|---|---|---|---|
| `social_role_kl` | Υπερ-εκπροσώπηση επαγγελμάτων | KL(κατανομή ρόλων του μοντέλου ‖ ILO-2023 baseline) πάνω στις 10 ISCO-08 κατηγορίες. Ο LLM τείνει να «γεννά» professionals πολύ πάνω από το ~11% της αγοράς. | 0.5 – 2.0 |
| `inter_agent_sim` | Ομοιογένεια πληθυσμού | Μέση pairwise TF-IDF cosine ομοιότητα μεταξύ **διαφορετικών** agents σε κοινή συζήτηση. Υψηλή = οι agents «ακούγονται ίδιοι». | 0.2 – 0.6 |
| `intra_agent_sim` | Επαναληπτικότητα | Μέση cosine μεταξύ **διαδοχικών** εκφράσεων του ΙΔΙΟΥ agent. Υψηλή = ο agent λέει ξανά και ξανά το ίδιο. | 0.2 – 0.6 |
| `keyword_persistence` | Primacy | Ποσοστό μεταγενέστερων εκφράσεων που κρατούν τη salient λέξη-κλειδί της πρώτης. Proxy για το «κολλάει στην πρώτη ιδέα». | 0.1 – 0.5 |
| `positivity` | Utopian/θετική στρέβλωση | (θετικές − αρνητικές λέξεις) / (θετικές + αρνητικές), στο [−1, 1]. Υψηλή = ρόδινη εικόνα του κόσμου. | 0.3 – 0.7 |

\* Ενδεικτικά, βασισμένα στα ευρήματα του Bian· **δεν είναι κατώφλια** — τα scores
παρουσιάζονται ως συνεχείς μετρικές, όχι pass/fail (βλ. ADR-006 §Deliberately not decided).

## Πώς τρέχει

```bash
python tools/bian_diagnostic.py --model qwen3:8b --out bian_scores_qwen3.json
```

- `--model` (υποχρεωτικό): το Ollama tag, ίδιο με το run σου.
- `--out`: αρχείο JSON (default `bian_scores_<model>.json`).
- `--n_samples`: δείγματα ανά probe (default 100 — spec του Bian: 100 role prompts, 20 dialogues × 6 agents). Χαμήλωσέ το για γρήγορο sanity check.
- `--url`: Ollama base URL (default `$OLLAMA_URL` ή `localhost:11434`).
- `--think {default,on,off}`: reasoning mode για μοντέλα που το υποστηρίζουν (qwen3). `default` = model default, `on`/`off` = force. **Το να τρέξεις ΚΑΙ τα δύο είναι από μόνο του diagnostic** (αλλάζει το CoT τα biases;) — π.χ. σε qwen3:8b (n=5) το thinking έριξε το `intra_agent_sim` 0.37→0.15 αλλά ήταν ~14× πιο αργό.
- `--force`: ξαναϋπολογίζει ακόμα κι αν υπάρχει το `--out` (αλλιώς **cached** — δεν ξανακαλεί το μοντέλο).

Έξοδος `bian_scores.json`:
```json
{
  "model": "qwen3:8b",
  "timestamp": "2026-07-21T...Z",
  "n_samples": 100,
  "scores": { "social_role_kl": 1.23, "inter_agent_sim": 0.41, "intra_agent_sim": 0.38,
              "keyword_persistence": 0.29, "positivity": 0.52 },
  "reference": "Bian et al. 2025, arXiv:2510.21180"
}
```

## Πότε να το τρέξεις

- **Πριν από κάθε νέα σειρά runs** με νέο ή διαφορετικό μοντέλο — για να έχεις καταγεγραμμένο το bias context του συγκεκριμένου μοντέλου δίπλα στα αποτελέσματα. Στον launcher, το `Include Bian bias scores` (Task 4.2) το τρέχει αυτόματα και αντιγράφει τα 5 scores στο `run_metrics.json config self-doc`, cached ανά μοντέλο.
- **Ως προβλέπτης του δικού σου finding:** αν ο population-level polarization συσχετίζεται με το `positivity` του μοντέλου, αυτό είναι το P28b («το bias του μοντέλου προβλέπει το συλλογικό αποτέλεσμα»).

## Ειδικές περιπτώσεις / σημειώσεις μεθόδου

- **Similarity stack:** ίδιο transparent TF-IDF + cosine με το `tools/semantic_analysis.py` (idf = log((1+n)/(1+df)) + 1, ώστε πανομοιότυπα κείμενα → cosine 1.0). Μπορεί να αντικατασταθεί με `nomic-embed` embeddings ως **documented condition** αργότερα — ο διαφανής baseline μένει για σύγκριση.
- **ILO baseline:** προσεγγιστικές παγκόσμιες μερίδες απασχόλησης ανά ISCO-08 major group (ILOSTAT-style, στρογγυλοποιημένες). Χρησιμεύουν ως σημείο αναφοράς για το *σχήμα* της απόκλισης, όχι ως ακριβή ποσοστά. Ο ρόλος ταξινομείται με keyword table (πρώτο match κερδίζει, high-skill groups πρώτα)· ασαφείς/άγνωστοι ρόλοι αγνοούνται στο KL.
- **Cross-language:** τα prompts είναι αγγλικά. Αν τεστάρεις μοντέλο σε άλλη γλώσσα, τα sentiment/keyword lexica θέλουν προσαρμογή — αλλιώς το `positivity`/`keyword_persistence` υποεκτιμώνται.
- **Καμία εξάρτηση** πέρα από stdlib (urllib) — offline-testable με fake `chat_fn` (βλ. `tests/test_bian_diagnostic.py`).

Spec: `docs/adr/ADR-006-diversification-scaffold.md` §Component 4· plan
`docs/superpowers/plans/2026-07-20-diversification-scaffold-plan.md` Task 4.1.
