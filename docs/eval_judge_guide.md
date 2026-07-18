# Οδηγός: Eval launcher & LLM judge

Αφορά τα `tools/eval_launcher.py` (GUI), `tools/eval_runs.py` (μετρικές/report) και `tools/judge_runs.py` (LLM κριτής). Κάθε πεδίο του GUI εξηγείται εδώ σε βάθος. Σύντομες εξηγήσεις υπάρχουν και μέσα στο παράθυρο.

---

## 0. Τα κουμπιά ενεργειών (τι κάνει το καθένα)

Όλα δουλεύουν πάνω στην ΙΔΙΑ λίστα runs (§1). Διαλέγεις runs μία φορά, μετά πατάς όποια ανάλυση θες.

- **Run comparison** → `eval_runs.py`. Η βασική ανάλυση: διαβάζει τις τροχιές, υπολογίζει B/D/P, convergence, holds/repairs/cleanups/fallbacks/leaks, agent-level bootstrap CIs, pairwise deltas, και γράφει `eval_report.md` + `aggregate.csv` (45 στήλες, §5) + figures. ΧΩΡΙΣ κλήσεις μοντέλου. Αυτό τρέχεις πρώτο.
- **Run judge** → `judge_runs.py` (θέλει Ollama). Ο LLM κριτής βαθμολογεί tweets/responses (stance match, closed-world leak, quality, direction support, persona consistency), cross-family, no-CoT, με context από τα ίδια τα αρχεία. Πάντα **Dry run** πρώτα για να δεις το prompt. §4.
- **Run semantic** → `semantic_analysis.py`. Μετρικές text-space που ΔΕΝ δίνει η −2..+2 κλίμακα: homogenisation (αν οι agents λένε τα ίδια), text convergence over time, stance separation (αν FOR/AGAINST διαβάζονται διαφορετικά), self-repetition. `tfidf` τρέχει τώρα, `ollama` για πραγματικά embeddings.
- **Inference emergence** → `inference_emergence.py`. Αρθρώνουν οι agents συμπεράσματα που ΔΕΝ γράφονται πουθενά στο corpus; Χρειάζεται το `.groundtruth.json` (πεδίο Multihop groundtruth). Κατώφλι auto-calibrated από τα null probes. §5β.
- **Scale info-loss** → `scale_infoloss.py`. Πόση κειμενική πληροφορία συμπιέζει η ακέραια κλίμακα: R² (variance explained), LOO text→rating recovery, flat-rating drift, distinct voices ανά rating. §5γ.
- **Build viewer(s)** → `build_viewer.py`. Ένα αυτόνομο offline HTML ανά run (δίκτυο + time scrubber + panel με tweets/replies ανά agent), δίπλα στο run. Θέλει run ΦΑΚΕΛΟΥΣ (loose CSV runs skip-άρονται).
- **Build A/B viewer** → `build_ab_viewer.py`. Τα ΠΡΩΤΑ ΔΥΟ runs της λίστας δίπλα-δίπλα σε έναν κοινό scrubber — για σύγκριση συνθηκών.
- **Master report** → `master_report.py`. Ομαδοποιεί ένα master CSV (που συσσώρευσες με Append, §3) ανά condition: mean ± std across runs (η πραγματική seed-to-seed διασπορά) + figure. Μία γραμμή ανά condition αντί για μία ανά run.

---

## 1. Runs to compare (η λίστα)

**Τι είναι «run».** Οτιδήποτε περιέχει ένα `*opinion_change*.csv`: είτε φάκελος run, είτε «χύμα» CSV (όταν πολλά runs έχουν γραφτεί στον ίδιο φάκελο, π.χ. thesis/ — κάθε τέτοιο CSV μετράει ως ξεχωριστό run και τα αδέρφια του αρχεία βρίσκονται από το κοινό prefix).

- **+ Add folder (run or parent):** διαλέγεις φάκελο. Αν είναι run, μπαίνει. Αν είναι γονικός, σκανάρει αναδρομικά και ανοίγει **checklist** με ό,τι βρήκε (filter box, Select all/none). Τα loose CSVs εμφανίζονται ως «loose run: …».
- **+ Add via files (multi-select):** native πολλαπλή επιλογή αρχείων (Ctrl/Shift-click). Διαλέγεις ένα οποιοδήποτε CSV μέσα από κάθε run που θες· ο γονικός φάκελος (ή το ίδιο το CSV αν είναι loose) γίνεται run.
- **condition label:** το όνομα της συνθήκης στο report (π.χ. strict / non-strict). Συμπληρώνεται αυτόματα από τη διαφορά των ονομάτων· διορθώνεται με το χέρι. Διπλότυπα labels παίρνουν αυτόματα `_2`, `_3` (αλλιώς θα κατέρρεαν σε μία συνθήκη).
- **Sort A-Z / Clear all / ✕:** τακτοποίηση της λίστας. Η λίστα θυμάται μεταξύ sessions.

## 2. Output

- **Report folder:** πού γράφεται το `eval_report/` (report + figures + aggregate.csv). Κενό = δίπλα στο πρώτο run.
- **Open report when done:** ανοίγει το markdown μόλις τελειώσει.
- **Open folder:** ανοίγει τον φάκελο εξόδου στον Explorer.

## 3. Aggregate CSV

Κάθε τρέξιμο γράφει **πάντα** φρέσκο `aggregate.csv` (snapshot της τρέχουσας σύγκρισης, 1 γραμμή/run, 45 στήλες — δες §5).

- **Append to master CSV:** επιπλέον, προσθέτει τις γραμμές σε ένα «master» αρχείο της επιλογής σου που **συσσωρεύει** runs across sessions. Μπορείς να έχεις διαφορετικά masters ανά πείραμα (π.χ. `v119_experiments.csv`)· το dropdown θυμάται τα 8 τελευταία.
- **Skip runs already present:** αν το ίδιο run (by name) υπάρχει ήδη στο master, δεν ξαναγράφεται. Ξετίκαρέ το μόνο αν θες σκόπιμα διπλές γραμμές.
- Αν το master έχει διαφορετικό header (παλιότερη έκδοση στηλών), το append αρνείται με καθαρό μήνυμα — δείξε νέο αρχείο.

## 4. LLM judge (πεδία)

- **Judge model:** ποιο τοπικό LLM βαθμολογεί. **Scan models** ρωτά το Ollama (`/api/tags`) και γεμίζει το dropdown με ό,τι είναι όντως εγκατεστημένο — τέρμα τα τυπογραφικά τύπου `qwenq2_2:8b`. Κανόνας: **άλλη οικογένεια** από τα μοντέλα των runs (qwen runs → llama judge), αλλιώς self-preference bias. Προτίμησε το δυνατότερο cross-family που σηκώνει το μηχάνημα.
- **Sample/run:** πόσα items (tweets+responses) βαθμολογούνται ανά run. Τυχαία επιλογή με **σταθερό seed**: ξανατρέξιμο = ίδια items (συγκρισιμότητα + cache). Περισσότερα = πιο αξιόπιστοι μέσοι όροι, περισσότερη ώρα. 120 καλό default για run 100 βημάτων (~200 items).
- **Context (off / light / full):** πόσο από το setup **του ίδιου του run** βλέπει ο κριτής — εξάγεται αυτόματα από τα αρχεία, δεν το πληκτρολογείς:
  - `off` — μόνο το κείμενο + generic ορισμός leak. Για γρήγορο έλεγχο.
  - `light` — + config line (world/RAG/version/model) + η **persona του συγγραφέα** (κάρτα + traits από το list_agent_descriptions.csv). Ενεργοποιεί και τη διάσταση persona_consistency.
  - `full` — + το **ακριβές κείμενο** των WORLD RULES (από το `EXTERNAL_WORLD_RULES` του main, με βάση τη στήλη World) + το **ακριβές Bias rule** (από το template της version). Χρησιμοποίησέ το στα closed-world runs: το leak κρίνεται πάνω στους ΠΡΑΓΜΑΤΙΚΟΥΣ κανόνες που δόθηκαν, όχι σε γενικόλογους.
- **Dry run:** χτίζει τα πάντα ΧΩΡΙΣ κλήσεις LLM και σώζει `example_prompt_<label>.txt` — το ανοίγεις και βλέπεις ακριβώς τι θα δει ο κριτής. Πάντα πρώτα dry run.
- **Explain flagged (two-pass rationale):** ΔΕΥΤΕΡΟ πέρασμα ΜΟΝΟ για τα items που ο κριτής μαρκάρισε leak=1: του ζητείται να ονομάσει, σε 1-2 προτάσεις, ΤΙ ακριβώς εξωτερικό υλικό επικαλείται το κείμενο. Κρίσιμη λεπτομέρεια: τρέχει ΑΦΟΥ τα σκορ έχουν κλειδώσει, άρα η εξήγηση δεν μπορεί να μολύνει την ετυμηγορία (το failure mode των CoT judges — βλ. roads_not_taken 1.2). Κόστος αμελητέο (μόνο τα flagged). Οι εξηγήσεις γράφονται στη στήλη `rationale` του `judge_scores_<label>.csv` — διάβαζέ τες όσο συμπληρώνεις το calibration sheet: αν ο judge ονομάζει «διαρροή» κάτι που ΔΕΙΧΤΗΚΕ, το βλέπεις αμέσως.
- **Calibration items:** N τυχαία items εξάγονται σε `calibration_<label>.csv` με κενές στήλες για **δικούς σου** βαθμούς. Τα βαθμολογείς με το χέρι και συγκρίνεις με τον judge· καλή συμφωνία (~90%) = τα σκορ είναι υποστηρίξιμα στη διπλωματική. Αυτό είναι το «human anchoring» (πρβλ. Μαλτέζος: 50 cases, 92%).

**Τι βαθμολογείται (διαστάσεις).** Ένα LLM call ανά item επιστρέφει ΟΛΕΣ τις διαστάσεις σε ένα JSON:
- tweets → `stance_match` 1-5 (ταιριάζει ένταση+κατεύθυνση με το rating; πιάνει τον ±1 που φωνάζει σαν ±2), `closed_world_leak` 0/1, `quality` 1-5, (`persona_consistency` 1-5 όταν υπάρχει persona).
- responses → `direction_support` 1-5 (**διφασικός έλεγχος σε μία κρίση**: η εξήγηση στηρίζει το συγκεκριμένο pre→post rating;), `closed_world_leak`, `quality`, (`persona_consistency`).

**Πώς κρίνεται κάθε run.** Κάθε run παίρνει **το δικό του context** (δικοί του κανόνες, δικό του claim, δικές του personas). Strict και open runs κρίνονται το καθένα πάνω στους δικούς του κανόνες — δεν υπάρχει κοινό μέτρο που να αδικεί το ένα.

**Outputs (στο `<report>/judge/`):** `judge_scores_<label>.csv` (item-level, για drill-down), `judge_summary_<label>.json` (μέσοι όροι), `judge_comparison.md` (ο συγκριτικός πίνακας — αυτό κοιτάς), `judge_cache_<label>.json` (κάθε κρίση αποθηκεύεται με SHA-256(model+prompt)· ξανατρέξιμο = 0 κλήσεις), `example_prompt_<label>.txt` + `example_rationale_prompt_<label>.txt` (από dry run), `calibration_<label>.csv`. Στο scores CSV: στήλη `rationale` για τα flagged items.

## 5. Τι σημαίνουν οι στήλες του aggregate.csv

Ταυτότητα: condition, run, model, version, world, rag, seed, steps, agents.
Αποτέλεσμα: `B/D/P_init`, `B/D/P_final`, `delta_B` (πόσο μετακινήθηκε ο μέσος), `extremity` (μέσο |γνώμη|), `frac_extreme` (% στους ±2), `entropy` (0=consensus, 1=πλήρης διασπορά), `camp_share` (μερίδιο μεγαλύτερου στρατοπέδου), `convergence_step` (από πότε σταθεροποιήθηκε το B).
Κίνηση: `total_changes`, `changed_agents`, `flips` (άλλαξαν πλευρά), `oscillations` (πάνω-κάτω αναποφασιστικότητα), `mean_drift`, `mean_abs_drift`.
Ροή/παρεμβάσεις: `listener_events`, `moves/up/down` + `move/up/down_rate`, `holds_total`+`hold_rate` (αναγκαστικά κρατήματα λόγω pipeline), `hard_repairs`+`repair_rate` (το pipeline ξαναέγραψε format), `soft_cleanups`+`cleanup_rate`, `step2_fallbacks`+`fallback_rate` (deterministic tweet αντί LLM), `empty_tweets`, `leak_step2/3` (validator-level closed-world warnings).

## 5β. Inference emergence (P8)

- **Τι μετράει:** αρθρώνουν οι agents συμπεράσματα που ΔΕΝ γράφονται σε κανένα απόσπασμα, μόνο υπονοούνται από αλυσίδες του multihop corpus; Κάθε tweet βαθμολογείται σε ομοιότητα προς κάθε `implied_conclusion` του groundtruth.
- **P8 groundtruth:** το `<corpus>.groundtruth.json` του multihop corpus (περιέχει τα κρυφά συμπεράσματα + τα null probes). ΔΕΝ το βλέπει ποτέ κανένας retriever/agent — μόνο αυτή η μέτρηση.
- **Κατώφλι (tau):** αυτο-βαθμονομείται ανά run: mean + 2σ της ομοιότητας των tweets προς τα null probes (ψευτο-συμπεράσματα που το corpus δεν στηρίζει, π.χ. «ο φούρναρης ήταν μπλεγμένος»). Ό,τι δεν ξεπερνά τον φούρναρη, δεν μετράει.
- **Backend:** ίδιο με το Semantic (tfidf τώρα — υποεκτιμά παραφράσεις· ollama για τα τελικά νούμερα).
- **Διαβάζεις:** max sim ανά αλυσίδα (και ΤΟ tweet που το πέτυχε — quoted στο report), emergence rate, first step, timeline figure. Runs που δεν είδαν το multihop corpus = negative control, δάπεδο ~0.20 (tfidf).

## 5γ. Scale info-loss (η ερώτηση «μας περιορίζει η −2..+2 κλίμακα;»)

- **Τι μετράει:** πόση από την ποικιλία του ΚΕΙΜΕΝΟΥ συμπιέζει το ακέραιο rating. R² = ποσοστό της
  text-space διακύμανσης που εξηγούν τα 5 σκαλιά· LOO accuracy = ανακτάται το rating μόνο από το
  κείμενο; (sign accuracy = τουλάχιστον η πλευρά;)· flat-rating drift = πόσο κινείται το κείμενο
  όσο ο ακέραιος μένει ίδιος· voices/rating = πόσες διακριτές «φωνές» συνυπάρχουν στο ίδιο σκαλί.
- **Χρήση:** λίστα runs + Backend (όπως το Semantic) → κουμπί «Scale info-loss». Χωρίς άλλα πεδία.
- **Πρώτα νούμερα (v119, tfidf):** R²≈0.16 αλλά LOO 0.85 / sign 0.90 — η κλίμακα κρατά τη ΣΤΑΣΗ
  σχεδόν πλήρως, χάνει όμως το ~85% της κειμενικής ποικιλίας (την επιχειρηματολογία). Γι' αυτό οι
  text-space μετρικές δεν είναι προαιρετικές. Με ollama backend τα νούμερα οριστικοποιούνται.

## 6. Ροή εργασίας (βήμα-βήμα)

1. `python tools\eval_launcher.py` → πρόσθεσε runs → labels.
2. **Run comparison** → διάβασε `eval_report.md` (αριθμητική σύγκριση).
3. **Dry run τικαρισμένο → Run judge** → άνοιξε `judge/example_prompt_<label>.txt`, δες ότι το context είναι σωστό.
4. Άνοιξε Ollama. Ξετίκαρε Dry run, βάλε Calibration items 25 → **Run judge**. Πρώτη φορά ~10-20′ για 2 runs × 120 items σε 8b judge· μετά cache.
5. Διάβασε `judge/judge_comparison.md`. Συμπλήρωσε το calibration CSV με το χέρι και σύγκρινε.
6. Θες κι άλλα runs αργότερα; Πρόσθεσέ τα στη λίστα και ξανατρέξε — ό,τι έχει κριθεί ξαναδιαβάζεται από το cache.

## 7. Συχνές παγίδες

- Judge ίδιας οικογένειας με τα runs → φουσκωμένα σκορ (self-preference). Πάντα cross-family.
- Μικρό sample (<50) → θορυβώδεις μέσοι. Ανέβασέ το πριν βγάλεις συμπέρασμα.
- Σύγκριση leak rates μεταξύ runs με ΔΙΑΦΟΡΕΤΙΚΟ world χρειάζεται προσοχή: κάθε run κρίνεται στους δικούς του κανόνες, άρα το «leak» δεν σημαίνει το ίδιο πράγμα.
- Τα RAG αποσπάσματα ανά interaction ΥΠΑΡΧΟΥΝ στο provenance log του run (`_retry_debug/rag_retrieval_*.csv`) και ο judge τα διαβάζει αυτόματα: κάθε item κρίνεται με το «MATERIAL EXPLICITLY SHOWN» του — αναφορά σε δειγμένο υλικό ΔΕΝ μετράει leak. (Αν το log λείπει, π.χ. πολύ παλιά runs, το leak κρίνεται σε επίπεδο διατύπωσης.)
