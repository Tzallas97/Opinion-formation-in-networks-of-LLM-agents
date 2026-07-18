# Multi-hop corpus spec — P8 (retrieval architecture as epistemic environment)

Προδιαγραφή για το corpus όπου η αρχιτεκτονική retrieval γίνεται πραγματική πειραματική μεταβλητή.
Ιδέα: κανένα απόσπασμα δεν περιέχει συμπέρασμα — μόνο ΑΛΥΣΙΔΕΣ αποσπασμάτων (συνδεδεμένες μέσω κοινών
οντοτήτων) υπονοούν κάτι. Τρεις retrievers στο ΙΔΙΟ corpus: lexical (baseline), dense (nomic-embed),
graph (BFS πάνω σε entity tags). Βλ. `docs/paper_pipeline.md` §P8 για το ερευνητικό σκεπτικό.

Δείγμα για review: `prompts/opinion_dynamics/Flache_2017/content/rag_corpus/multihop_v119_sample.jsonl`
(+ `.graph.json`, `.groundtruth.json`).

---

## 1. Αρχεία και όρια εμπιστοσύνης (τι επιτρέπεται να δει ο καθένας)

| Αρχείο | Περιεχόμενο | Ποιος το διαβάζει |
|---|---|---|
| `<name>.jsonl` | Τα αποσπάσματα: `id, text, direction, topic_key, topic_name, entities` | Και οι τρεις retrievers (ο lexical/dense ΜΟΝΟ το text· ο graph και τα entities) |
| `<name>.graph.json` | Alias map οντοτήτων: canonical → surface forms | Μόνο ο graph retriever (seed matching) |
| `<name>.groundtruth.json` | Αλυσίδες (μέλη, κατεύθυνση, implied conclusion), ρόλοι (chain/standalone/distractor), test queries | ΜΟΝΟ η αξιολόγηση (Στάδιο Α benchmark, inference-emergence metric). ΠΟΤΕ retriever ή prompt. |

Γιατί έτσι: αν `chain_id`/`implied_conclusion` ήταν ορατά στον retriever, θα «σερβίραμε την απάντηση»
και η σύγκριση θα ήταν στημένη. Το jsonl διαβάζεται από το υπάρχον `_load_simple_rag_corpus`
(JSONL branch)· από το main integration (2026-07-17) το `entities` ΠΕΡΝΑΕΙ από τον loader (το
χρειάζεται ο graph backend) αλλά **δεν φτάνει ποτέ σε prompt**: το rendering χρησιμοποιεί μόνο
`text` + direction labels. Η αορατότητα των tags επιβάλλεται στο όριο του rendering, όχι του
loading. Chains/conclusions μένουν σε ξεχωριστό αρχείο που ΚΑΝΕΝΑ κομμάτι του main δεν διαβάζει.

## 2. Σχήμα

Γραμμή jsonl (πεδία πέρα από τα υπάρχοντα: μόνο `entities`):

```json
{"id": "mh_c1_01", "text": "A 1999 municipal filing shows structural engineer D. Harris signed the final integrity certificate for the North Tower's steel frame.", "direction": "supportive", "topic_key": "v119", "topic_name": "Twin towers conspiracy claim", "entities": ["D. Harris", "North Tower"]}
```

`graph.json`:

```json
{"entities": {"North Tower": {"aliases": ["north tower", "twin towers", "the tower", "towers", "tower"]}, "D. Harris": {"aliases": ["d. harris", "harris"]}}}
```

`groundtruth.json`:

```json
{"chains": [{"chain_id": "C1", "members": ["mh_c1_01", "mh_c1_02", "mh_c1_03"], "direction": "supportive", "implied_conclusion": "..."}],
 "roles": {"mh_c1_01": "chain", "mh_s_01": "standalone", "mh_d1_01": "distractor"},
 "test_queries": [{"qid": "q1", "text": "...", "expected_chains": ["C1", "C2"]}]}
```

## 3. Κανόνες συγγραφής (αυτοί κάνουν το πείραμα έγκυρο)

1. **Κανένα απόσπασμα δεν διατυπώνει το συμπέρασμα** της αλυσίδας του, ούτε μερικώς. Κάθε μέλος
   διαβάζεται μόνο του ως αθώα/βαρετή καταγραφή (register: έγγραφα — filings, logs, notices,
   procurement records — όχι επιχειρηματολογία).
2. **Ενιαίο register παντού.** Και τα standalones του multihop corpus γράφονται σαν «έγγραφα».
   Το υπάρχον v119 corpus (discourse register: «supporters argue…») ΔΕΝ αναμιγνύεται — αλλιώς το ύφος
   γίνεται confound της αρχιτεκτονικής. (Απορριφθείσα εναλλακτική: reuse των 30 υπαρχόντων ως standalones.)
3. **Anti-leak λεξιλογίου:** τα μέλη μιας αλυσίδας ΔΕΝ μοιράζονται χαρακτηριστικό λεξιλόγιο πέρα από
   τα ονόματα των κοινών οντοτήτων· και τα ενδιάμεσα/ακραία μέλη αποφεύγουν τις λέξεις του claim
   (π.χ. το mh_c1_03 λέει «the site's debris-removal contract», όχι «the tower's»). Αλλιώς ο lexical
   βρίσκει την αλυσίδα από τη διατύπωση και η σύγκριση αδειάζει. (Γι' αυτό απορρίφθηκε και η
   template-παραγωγή αλυσίδων: κοινή φρασεολογία template = δώρο στον lexical.)
4. **Κάθε tagged οντότητα εμφανίζεται αυτολεξεί** (μέσω κάποιου alias, case-insensitive) στο text του
   αποσπάσματος. Έτσι μένει ανοιχτός δρόμος για μελλοντική συνθήκη «αυτόματη εξαγωγή οντοτήτων» πάνω
   στο ίδιο corpus.
5. **Μήκος αλυσίδων 2-3 hops** (3-4 μέλη το πολύ σε εξαιρέσεις). Ρηχές αλυσίδες = δίκαιο τεστ και για
   8b agents· η ένωση Α+Β⇒Γ είναι το μετρούμενο, όχι σπαζοκεφαλιά.
6. **Ισορροπία κατευθύνσεων:** ίσος αριθμός αλυσίδων supportive και criticism, συγκρίσιμα μήκη·
   standalones επίσης ισορροπημένα. Καμία πλευρά δεν κερδίζει επειδή «έχει τα multi-hop».
7. **Distractors:** αλυσίδες συνδεδεμένες σε πραγματικές οντότητες των κανονικών αλυσίδων που
   καταλήγουν σε αδιέξοδο (χωρίς conclusion). Ο graph ΘΑ τις φέρνει πού και πού — by design: φέρνει
   «συνδεδεμένο», όχι «χρήσιμο» υλικό· ο agent κρίνει. Χωρίς αυτές, graph = μαντείο = άκυρη σύγκριση.
8. **Fictional οντότητες** (πρόσωπα/εταιρείες/εργαστήρια) — καμία σύμπτωση με πραγματικά πρόσωπα.
   Το claim του v119 μένει ως έχει (είναι το αντικείμενο του run).
9. **Direction των chain members = η κατεύθυνση της αλυσίδας τους** (απόφαση: το μέλος υπάρχει για να
   υπηρετήσει το επιχείρημα, κι ας είναι μόνο του ουδέτερο· εναλλακτική «όλα context» απορρίφθηκε
   γιατί θα αδειάζε το balanced mode από chains). Distractors → `context`.
10. **Μέγεθος πλήρους corpus (v1):** 3 αλυσίδες/πλευρά × 3 μέλη ≈ 18 chain snippets, 3-4 distractor
    αλυσίδες ≈ 8, ~30-35 standalones → **~60 σύνολο**, αναλογία 2:2:1 supportive:criticism:context
    όπως το υπάρχον corpus.

## 4. Οι τρεις retrievers (συμβόλαια v1)

Κοινά: ντετερμινιστικοί (ίδιο query → ίδιο αποτέλεσμα), top-k=4 (όπως το τωρινό `rag_top_k`),
provenance logging όπως σήμερα (`rag_retrieval_*.csv` — παίρνει στήλη `retriever`).

- **lexical** = ο υπάρχων simple backend, ανέγγιχτος. Διαβάζει μόνο text.
- **dense** = embeddings (nomic-embed-text μέσω Ollama, on-disk cache όπως judge/semantic)·
  cosine(query, snippet), top-k. Διαβάζει μόνο text. Offline προϋπολογισμός των snippet vectors.
- **graph** = (α) build: inverted index entity→snippets από τα tags· ακμή snippet↔snippet όταν
  μοιράζονται οντότητα **με degree ≤ hub_max=6** — οι hub οντότητες (π.χ. North Tower, degree 10)
  είναι topic anchors που ΣΠΕΡΝΟΥΝ την αναζήτηση αλλά δεν μετράνε ως κρίκοι, αλλιώς όλα τα chain
  heads γίνονται κλίκα και το «μονοπάτι» φιδοσέρνεται στα heads (παθολογία που έπιασε το Στάδιο Α).
  (β) seed: σάρωση query για aliases (longest-match, word boundaries, span consumption· σε
  αμφισημία — π.χ. σκέτο «Harris» — κερδίζει η πρωτεύουσα οντότητα του alias map). (γ) BFS
  entity→snippet→entity, βάθος ≤3 hops (οι 4-μελείς αλυσίδες είναι 3-hop). (δ) επιλογή v2: best
  συνεχόμενο μονοπάτι με score = (**stopword-filtered query overlap** των κειμένων του, μήκος),
  tie-break μικρότερη ακολουθία ids — το query διαλέγει ΠΟΙΑ αλυσίδα, όχι η τοπολογία· χωρίς το
  stopword filter, το κοινό «the» επιδοτεί +1/κόμβο τα μακριά μονοπάτια (δεύτερη παθολογία του
  Σταδίου Α). Γέμισμα υπολοίπων κατά BFS (depth, id). (ε) rendering με σειρά μονοπατιού.
  Fallback χωρίς seeds: top-1 lexical snippet → οι οντότητές του γίνονται seeds.
  Υλοποίηση: `tools/retrievers.py` (και οι τρεις)· benchmark: `tools/chain_benchmark.py`.

Για την τριπλή σύγκριση: `rag_content_mode=full` σε ΟΛΟΥΣ (το balanced direction-quota θα έσπαγε
τις αλυσίδες και θα μασκάρειε την αρχιτεκτονική)· η ισορροπία είναι ιδιότητα του corpus (κανόνας 6).

## 5. Στάδιο Α — chain-recovery benchmark (χωρίς LLM, τρέχει τώρα)

Πάνω στα `test_queries` του ground truth: για κάθε retriever × query, **chain recovery@k**,
full-chain rate, best-chain (per query) και distractor share. Εργαλείο: `tools/chain_benchmark.py`.

**Αποτελέσματα v1 (2026-07-17, k=4, max_hops=3):** lexical mean recovery 0.25, full-chain 0.00
σε ΟΛΑ τα queries (φέρνει θραύσματα, όπως προβλέφθηκε)· **graph best-chain full 7/7**, mean
recovery 0.58, distractor share 0.07 (φέρνει το bait στα q2/q6 — το σχεδιασμένο «συνδεδεμένο ≠
χρήσιμο»)· dense εκκρεμεί (θέλει τοπικό Ollama μία φορά — cache μετά). Report:
`rag_corpus/benchmark/chain_benchmark.md`. Χρειάστηκαν δύο διορθώσεις ranking (hub-clique,
stopword length subsidy) — και οι δύο πιάστηκαν offline, μηδέν compute κάηκε: αυτός είναι ο λόγος
ύπαρξης του Σταδίου Α.

**Το φρένο (gate):** αν ο graph δεν υπερτερεί ΚΑΘΑΡΑ στο full-chain rate offline, το corpus
ξαναδουλεύεται ή το P8 επιστρέφει στο ράφι. **Ετυμηγορία 2026-07-17: PASS** (0.58/1.00 vs
0.25/0.00 σε mean recovery / best-chain full)· επικύρωση και με dense όταν τρέξει τοπικά.

## 6. Στάδιο Β — τα runs (PENDING μέχρι να υπάρξει compute)

Ίδιο config/seed/personas/steps, εναλλαγή ΜΟΝΟ retriever (3 συνθήκες), corpus σταθερό. Μετρικές:
ό,τι ήδη έχουμε (B/D/P, transitions, holds, judge full-context, semantic) **+ inference emergence**
— υλοποιημένο: `tools/inference_emergence.py` (και κουμπί στο eval GUI, Semantic section). Για κάθε
tweet: similarity προς κάθε `implied_conclusion` (tfidf τώρα / ollama embeddings για τα τελικά)·
αναφέρει per chain: max sim + το ίδιο το κοντινότερο tweet (ποιοτικός έλεγχος), emergence rate,
first step· timeline figure. Κατώφλι **αυτο-βαθμονομημένο**: τα distractors απέκτησαν `null_probe`
κείμενα (εύλογα-ακουγόμενα συμπεράσματα που το corpus ΔΕΝ στηρίζει, π.χ. «ο φούρναρης ήταν
μπλεγμένος») και tau = mean + 2σ της ομοιότητας όλων των tweets προς αυτά. **Negative control
(2026-07-17):** τα δύο παλιά v119 runs (που δεν είδαν ποτέ το multihop corpus) πιάνουν max sim
0.10-0.20 με tfidf — αυτό είναι το δάπεδο θορύβου από το κοινό λεξιλόγιο του claim· το Στάδιο Β
ψάχνει graph-τιμές καθαρά πάνω από το δάπεδο ΚΑΙ πάνω από τις lexical-τιμές του ίδιου corpus.
Αν ο graph-πληθυσμός αρθρώνει τα άγραφα συμπεράσματα και ο lexical όχι — αυτό είναι το εύρημα του P8.

## 7. Integration στον main — ✅ ΕΓΙΝΕ (2026-07-17)

`--rag_backend off|simple|dense|graph` (+ `--rag_embed_model`, default nomic-embed-text)·
ο simple ανέγγιχτος ως baseline. Ενιαίο σημείο διακλάδωσης μέσα στο `_rag_context_for_interaction`
(ό,τι βλέπει το prompt μένει πανομοιότυπο)· **fail-fast** `_init_rag_architecture()` στην εκκίνηση
(λείπει `.graph.json` / πεσμένο Ollama / άδειο corpus → καθαρό SystemExit ΠΡΙΝ τρέξει βήμα)·
dense cache: `<corpus>.embcache.json` δίπλα στο corpus· provenance log: νέα στήλη `retriever`.
GUI: dropdown off/simple/dense/graph + muted label· οδηγός: `docs/ui_launcher_guide.md` §4.
Επαλήθευση χωρίς live LLM: py_compile ΟΚ, AST-extracted functional test (graph επιστρέφει την C1
αλυσίδα + bait μέσα από το wiring του main· dense/graph fail-fast μηνύματα σωστά), launcher
tkinter-stub instantiation ΟΚ. Live smoke: PENDING #9.

## 8. Σειρά εργασιών

1. ✅ Spec + δείγμα 15 αποσπασμάτων για review. _(2026-07-17)_
2. ✅ Review: επιβεβαιώθηκε η ιδιότητα «μόνα τους αδιάφορα, συνδεδεμένα ιστορία». _(2026-07-17)_
3. ✅ Πλήρες corpus v1: `multihop_v119.jsonl` (60 αποσπάσματα — 6 αλυσίδες 3σ/3κ με hops [2,2,3],
   3 distractor αλυσίδες, 34 standalones· 24/24/12 supportive/criticism/context· το δείγμα είναι
   υποσύνολό του) + `tools/corpus_lint.py` (E1-E5 errors: πεδία/ids/tag↔text αμφίδρομα με span
   consumption/ground-truth συνέπεια/reachability· W1-W4 warnings: anti-leak vocab, claim vocab,
   ισορροπία, hub degrees). Lint: 0 errors, 2 αποδεκτά warnings (κοινές topical λέξεις σε μη
   γειτονικά μέλη). Loader-compat επαληθευμένο (60/60 ορατά, entities αόρατα στη sim). _(2026-07-17)_
4. ✅ Retrievers (`tools/retrievers.py`) + Στάδιο Α benchmark (`tools/chain_benchmark.py`) —
   gate PASS για lexical vs graph· η dense στήλη θέλει ένα τοπικό τρέξιμο με Ollama
   (`ollama pull nomic-embed-text` → ξανατρέξε το benchmark — τα embeddings κασάρονται). _(2026-07-17)_
5. Gate του Σταδίου Α → ΠΕΡΑΣΕ: επόμενο = main integration (`--rag_backend dense|graph`) +
   PENDING run entry (Στάδιο Β).
