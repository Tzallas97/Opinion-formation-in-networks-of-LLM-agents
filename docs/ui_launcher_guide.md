# Οδηγός: ui_lancher11 — κάθε ρύθμιση και οι ειδικές περιπτώσεις

Ο launcher μαζεύει όλες τις ρυθμίσεις ενός πειράματος, χτίζει το command line και τρέχει το `opinion_dynamics_test_network_qwen.py`. Αυτό το αρχείο εξηγεί ΤΙ κάνει κάθε ρύθμιση, ποιες τιμές έχει, και τις παγίδες. Θα επεκτείνεται όποτε αγγίζουμε καινούριο κομμάτι.

> Γρήγορος κανόνας για το τι τρέχουμε συνήθως: main script = test_network_qwen από τον launcher, `free_bounded` update mode, step3 πάντα με thinking, closed world strict RAG, strictness ως τεκμηριωμένη συνθήκη.

---

## 1. Scripts & μοντέλα

- **Script:** ποιο main θα τρέξει. Πρακτικά μόνο το `opinion_dynamics_test_network_qwen.py` δουλεύει από εδώ· το `opinion_dynamics_v3_check.py` (calibration) τρέχει από terminal.
- **Step2 model / Step3 model:** ποιο LLM γράφει τα tweets (step2) και ποιο αποφασίζει την ενημέρωση γνώμης (step3). Μπορούν να διαφέρουν. `none` = χωρίς LLM σε εκείνο το βήμα (για ελέγχους).
- **Think mode:** αν το μοντέλο θα «σκέφτεται» (chain-of-thought tokens) πριν απαντήσει. Ξέρει ανά μοντέλο αν υποστηρίζεται (registry: qwen3/qwq/deepseek-r1 = ναι· άγνωστα μοντέλα = επιτρεπτό/άγνωστο, δεν γκριζάρει). **Ειδική περίπτωση:** το step3 κρατά ΠΑΝΤΑ thinking — αν το output κοπεί, δεν κόβουμε τη σκέψη, ανεβάζουμε το num_predict (boost σε ≥8192) και ξαναζητάμε.
- **num_predict:** μέγιστο πλήθος tokens απάντησης. Στα thinking μοντέλα η σκέψη ΚΑΤΑΝΑΛΩΝΕΙ από αυτό το budget — μικρό num_predict + thinking = κομμένα/άδεια outputs. Εκεί οφείλονταν τα περισσότερα άδεια tweets.

## 2. Prompt version (topic + bias)

- **Version set:** φάκελος prompt (σκανάρεται αυτόματα από το prompts/). Το όνομα κωδικοποιεί: **θέμα** (v52 = Σελήνη, v119 = 9/11, v130 = simulation hypothesis), **confirmation bias** (σκέτο / `confirmation_bias` / `strong_confirmation_bias`) και **κατεύθυνση claim** (`_reverse` = το claim διατυπωμένο από τη σκοπιά της συνωμοσίας).
- **Bias (μέσα στο template):** none = καμία αντίσταση· weak = προτιμά ό,τι συμφωνεί· strong = σχεδόν μόνο ό,τι συμφωνεί, **ασύμμετρο**: εύκολη κίνηση προς την πλευρά που ήδη κλίνει, πολύ δύσκολη η επιστροφή. Δεν αναφέρεται ποτέ μέσα στο output του agent.
- Το ακριβές claim γράφεται μέσα στα templates (`CLAIM: ...`).

## 3. World (κλειστός/ανοιχτός κόσμος)

- **World:** ποιοι κανόνες πληροφορίας ισχύουν. `closed*` = ο agent «είναι μόνος στο δωμάτιό του», απαγορεύεται εξωτερική γνώση· μόνο ό,τι του δείχνεται (tweet, persona, RAG/fact-pack αποσπάσματα). `closed_strict_rag` = η αυστηρή εκδοχή που τρέχουμε. `open*` = επιτρέπεται η γνώση του μοντέλου (τα πραγματολογικά θέματα τότε συγκλίνουν στη σωστή απάντηση από μόνα τους — γι' αυτό τα πειράματα κοινωνικής δυναμικής θέλουν closed).
- Το ακριβές κείμενο κανόνων ζει στον main (`EXTERNAL_WORLD_RULES[world]`) και μπαίνει στο prompt ως {WORLD_RULES}. **Η διατύπωση είναι πειραματική μεταβλητή** — μην την αλλάζεις σιωπηλά μεταξύ runs.
- **Closed-world enforcement είναι SOFT:** παραβίαση στο output καταγράφεται ως warning (val_warnings, metrics `closed_world_leak_*`), ΔΕΝ γίνεται retry. Απόφαση: μετράμε τη διαρροή, δεν την κρύβουμε.

## 3β. Solo check (ρώτα το μοντέλο ΜΟΝΟ ΤΟΥ)

- **Τι κάνει:** `on` = καμία persona, κανένα δίκτυο, καμία αλληλεπίδραση — κάθε «agent» είναι ένα
  ανεξάρτητο δείγμα του μοντέλου που απαντά στο `step1_report.md` για το claim (της version, με
  fallback στο κοινό `llm_check_template`). Μετράει το **prior του ίδιου του μοντέλου** για το
  θέμα — το baseline του cross-model benchmark (PENDING #7): «τι πιστεύει το μοντέλο πριν το
  βάλουμε σε κοινωνία».
- **Γιατί μέσα στον main (και όχι το παλιό v3_check):** ίδιο native inference μονοπάτι, ίδια
  decoding options, config self-doc στο metrics JSON → τα solo νούμερα συγκρίνονται ευθέως με τα
  run νούμερα. Το παλιό `opinion_dynamics_v3_check.py` (ξεχωριστό langchain stack) αποσύρθηκε σε
  deprecated wrapper.
- **Έξοδοι:** τροχιές `*_opinion_change_*.csv` (ίδιο σχήμα με τα runs — τα διαβάζει το eval GUI),
  raw responses CSV, `*_solo_check_metrics.json` (config + native_calls/parse_failures/reprompts,
  τελική κατανομή). Κανόνας ενημέρωσης ανά βήμα: delta clamp ±1 από ουδέτερη αρχή (όπως το v3).
- **Προσοχή:** όσο είναι `on`, τα πεδία δικτύου/RAG/persona αγνοούνται· δεν ισχύει ο περιορισμός
  «agents πολλαπλάσιο του 5».
- **UI gating (2026-07-17):** με solo `on` τα άσχετα sections γκριζάρουν αυτόματα (Global
  constraints, Step-2/Step-3 overrides, RAG, True-open, Network, Personas, JSON profiles) και
  το Steps απενεργοποιείται — τα δείγματα = Agents. Ενεργά μένουν: script, μοντέλο, prompt
  version/variant, seed, agents, LLM defaults (decoding), out, multiple runs (πολλαπλά seeds =
  πολλαπλές ανεξάρτητες δειγματοληψίες του prior). Με `off` επανέρχονται οι προηγούμενες
  καταστάσεις και ξανατρέχουν οι υπό συνθήκη κανόνες γκριζαρίσματος.
- **Ιδέα σε αναμονή (ROADMAP):** ειδικό live view για solo runs — δεν υπάρχει αρχική κατανομή
  γνωμών ώστε να δείχνει distribution· ταιριάζει προβολή «running tally των N απαντήσεων».

## 4. RAG & Fact pack (τα στοιχεία)

- **RAG backend:** η ΑΡΧΙΤΕΚΤΟΝΙΚΗ της ανάκτησης — από το P8 και μετά είναι πειραματική μεταβλητή, όχι υδραυλικά. `off` = καμία ανάκτηση. `simple` = λεξιλογική επικάλυψη (bag-of-words) — το διαφανές baseline μας, ό,τι έτρεχε πάντα. `dense` = σημασιολογική ομοιότητα με embeddings (Ollama· κάνε πρώτα `ollama pull nomic-embed-text`· τα διανύσματα κασάρονται σε `<corpus>.embcache.json` δίπλα στο corpus — αν το Ollama είναι κλειστό, το run σταματά ΣΤΗΝ ΕΚΚΙΝΗΣΗ με καθαρό μήνυμα, όχι στη μέση). `graph` = ανάκτηση αλυσίδων μέσω γράφου οντοτήτων — θέλει multihop corpus με entity tags + `<corpus>.graph.json` (alias map) δίπλα του· αλλιώς σταματά στην εκκίνηση. Για την τριπλή σύγκριση P8: ίδιο corpus (`multihop_v119.jsonl`), ίδιο seed, εναλλαγή ΜΟΝΟ backend, content mode = `full` (τα quotas του balanced θα έσπαγαν τις αλυσίδες — στο dense/graph τα content-mode quotas ΔΕΝ εφαρμόζονται έτσι κι αλλιώς: καθαρό top-k, η ισορροπία είναι ιδιότητα του corpus). Το provenance log κάθε run γράφει πλέον και στήλη `retriever`. Βλ. `docs/multihop_corpus_spec.md`.
- **RAG content mode:** ποια αποσπάσματα από το τοπικό corpus δείχνονται στον listener στη φάση κρίσης: `supportive` (υπέρ του claim), `criticisms` (κατά), `neutral`, `balanced` (μείγμα), `full`, `off` (καθαρά κοινωνική επιρροή). Στο `simple` backend το retrieval = **λεξιλογική επικάλυψη**, ντετερμινιστικό με seed, k μικρό (~4). Σκόπιμα ΟΧΙ νευρωνικό ως baseline — διαφάνεια/αναπαραγωγιμότητα, απομονώνει την κατεύθυνση των στοιχείων από την ποιότητα retriever. (Ισχύει ΜΟΝΟ για το simple· dense/graph = καθαρό top-k.)
- **Fact pack mode:** ξεχωριστό, επιμελημένο σετ στοιχείων ανά θέμα (`off/full/supportive/criticisms/contextual`). Στο closed world, RAG+fact pack είναι σχεδόν η ΜΟΝΗ εξωτερική πληροφορία — γι' αυτό έχουν δυσανάλογα ισχυρή επίδραση (μπορούν να αναποδογυρίσουν και hubs).
- **Παγίδα A/B:** αν συγκρίνεις δύο runs, κράτα ΙΔΙΟ rag mode — τα δύο v119 runs μας διέφεραν και σε strictness ΚΑΙ σε rag (balanced vs full) → confound.

## 5. Ενημέρωση γνώμης (update model)

- **Allowed update mode:** `assimilation_only` = μόνο προς τη γνώμη του ομιλητή (Flache assimilation)· `free_bounded` = ελεύθερη κατεύθυνση μέσα σε όρια. **Εμείς τρέχουμε free_bounded.**
- **max_step_change = 1:** μία θέση ανά αλληλεπίδραση το πολύ. Το ALLOWED_FINAL_RATING_SET στο prompt βγαίνει από αυτό.
- **same_side_edge_unlock:** ξεκλείδωμα άκρων όταν ομιλητής/ακροατής είναι ίδια πλευρά. **ΕΙΔΙΚΗ ΠΕΡΙΠΤΩΣΗ: κάτω από free_bounded είναι NO-OP** (ο κώδικας επιστρέφει τα vals ως έχουν) — άσ' το 0, δεν κάνει τίποτα.
- Εκτός κλίμακας ratings διορθώνονται στο κοντινότερο επιτρεπτό (`_resolve_invalid_step3_rating`) και μετριούνται.

## 6. Validation strictness & rescue

- **Validation strictness:** `strict` = οι content validators (ύφος/δομή/κατεύθυνση) μπορούν να απορρίψουν και να ξαναζητήσουν output· `warn_only` = όλα καταγράφονται αλλά τίποτα δεν ξαναζητιέται· `format_only` = μόνο έλεγχος format (FINAL_RATING/TWEET parsing). **Οι world-model έλεγχοι (closed world, v130 topic claims) μένουν πάντα ενεργοί ως warnings σε κάθε mode.**
- **Γιατί μας νοιάζει:** το strictness επηρεάζει το αποτέλεσμα (στο A/B μας άλλαξε ποιοτικά την εικόνα: κίνηση vs πόλωση/deadlock). Γι' αυτό είναι **συνθήκη προς αναφορά**, όχι «τεχνική λεπτομέρεια». Υπερβολικά αυστηροί validators ομογενοποιούν τα μοντέλα → χάνεται discriminative power.
- **Wrong-side explanation re-query:** on = όταν η εξήγηση δείχνει λάθος πλευρά/κομμένη/leak, γίνεται ΜΙΑ επιπλέον κλήση που ξαναγράφει ΜΟΝΟ την εξήγηση — **το rating μένει όπως δόθηκε** (δεν αγγίζεται). Off = deterministic fallback εξήγηση.
- **Deterministic:** temperature 0 / top_k 1 / top_p 1 — αναπαραγώγιμα runs, λιγότερη ποικιλία. Χρήσιμο για debugging/σύγκριση, όχι για «φυσικά» κείμενα.
- **Structured output (EXPERIMENTAL, default off):** on = τα step2/step3 native calls ζητούν **grammar-constrained JSON** από το Ollama (`format=json`, με σχετική οδηγία στο prompt: `{"final_rating": int, "tweet"|"explanation": str}`) και η απάντηση μετατρέπεται πίσω σε κανονικό `FINAL_RATING/TWEET|EXPLANATION` κείμενο ΠΡΙΝ από κάθε parsing. Γιατί: το μοντέλο ΔΕΝ μπορεί να βγάλει χαλασμένο format, άρα τα post-hoc format repairs (και το selection bias τους) συρρικνώνονται — καθαρότερη σύγκριση μοντέλων. Αποτυχίες μετατροπής μετρώνται στο metrics (`structured_output_parse_fail_total` / `converted_total`). Προσοχή: σε thinking μοντέλα το format constraint μπορεί να περιορίσει το reasoning output· δήλωσέ το ως συνθήκη και κράτα off στα baseline runs. Δεν έχει τρέξει live ακόμα — βλ. PENDING_RUNS #6.
- **Allow silence in step2 (ADR-006 §Component 2, default off):** on = το step2 prompt αποκτά mechanism-agnostic silence-option block· ο agent μπορεί να επιστρέψει το token `<silent>` αντί για tweet. off = byte-identical με προ-ADR-006 (marker + surrounding blank lines strippάρονται από τον renderer). **Πότε να το ενεργοποιήσεις:** για P12 spiral-of-silence στα LLM populations — το paper contribution είναι η διάκριση **communicative silence** (αγοράζει preference falsification, Kuran 1995) από **opinion silencing** (τι μετρά ο Zhong 2025 arXiv:2510.02360). **Καβεάτ (2026-07-20):** το scaffold προσθέτει flag + renderer + parser as importable module (`scripts/step2_io.py`). Ο **runtime dispatch** στο step2 pipeline (silence counter, refusal counter, `_silence_log.csv` με 8 στήλες context, `tools/silence_mechanism.py` post-hoc disentangler για 4 candidate mechanisms) είναι Task 2.3b/2.4 του ADR-006 και ρητά deferred μέχρι P12 pilot planning. Με το flag on αλλά χωρίς dispatch, ένα `<silent>` response από το μοντέλο πέφτει στο υπάρχον parse-fail path. Reference: Noels 2025 arXiv:2504.03803 §Fig.1 για τη hard-refusal ταξινομία που ο parser διακρίνει από την επιλεγμένη σιωπή.

## 7. Personas (4+1 στρώματα)

- **Use personas / JSON profiles vs manual:** δύο δρόμοι — έτοιμα JSON profiles (persona_profiles.json) ή manual τιμές ανά μεταβλητή. (Εκκρεμεί mode-selector που να γκριζάρει το ανενεργό.)
- **Profile label (6 αρχέτυπα):** Institutionally trusting pragmatist, Suspicious anti-institutional skeptic, Open-minded skeptical reviser, Authority-leaning stabilizer, Uncertainty-tolerant agnostic, Low-trust practical realist — γεμίζουν τα 5 core με συνεπείς συνδυασμούς. Custom = δικό σου όνομα.
- **Core causal (γενικά αιτιακά):** `institutional_trust` (εμπιστοσύνη σε θεσμούς — κατευθυντικό φίλτρο αποδοχής στοιχείων), `official_narrative_suspicion` (καχυποψία στην επίσημη αφήγηση), `uncertainty_tolerance` (αντοχή στο «δεν ξέρω»), `evidence_style` (concrete/source/coherence/intuition/mixed — τι είδους επιχείρημα πείθει), `openness_to_update` (πόσο εύκολα μετακινείται). 5-βάθμιες κλίμακες (Likert-style).
- **Core causal — ADR-006 §Component 1 προσθήκες (2026-07-20):** `locus_of_control` (internal/mixed/external — πώς αλλάζει γνώμη: από argument/evidence ή από majority pressure· default `mixed` = μηδέν πρόσθετη γραμμή στην κάρτα, Rotter 1966) και `contrarianism` (very_low → very_high — προδιάθεση απόκλισης από την πληθυσμιακή πλειοψηφία· default `medium` = ουδέτερο, Flache 2017 repulsive influence). **Πότε να τα αλλάξεις:** για P4 persona-ablation runs που ρωτούν ποιες διαστάσεις καθορίζουν συλλογικά outcomes· για intervention studies (P29/P10 event_injection) που ρωτούν αν internal-LoC ανταποκρίνεται σε argument-based shocks ενώ external-LoC σε majority-shift· για echo-chamber studies (P21/P30) που ρωτούν αν contrarians σπάνε την κλιμάκωση σε clustered networks. Με defaults `mixed/medium` κάθε προϋπάρχον run είναι byte-identical (σκόπιμο).
- **Expressive / debate style (validator permissions — ΔΕΝ αλλάζει πεποιθήσεις):** `value_orientation` (procedural/outcome_focused → επιτρέπει συνεπειοκρατικό framing), `agency_vs_fatalism` (fatalistic → επιτρέπει «δομικές δυνάμεις» framing), `conflict_style` (combative → επιτρέπει κοφτό τόνο). **Off by default** — άναψέ τα μόνο για εκφραστική ποικιλία ή persona-adherence tests. Χαλαρώνουν τους validators ώστε in-character γλώσσα να μη μαρκάρεται ως λάθος.
- **Topic causal (ανά θέμα, 3-4 traits):** v52: cold_war_motive_sensitivity, engineering_evidence_weight, visual_anomaly_sensitivity· v119: geopolitical_motive_sensitivity, conspiracy_coordination_prior, anomaly_sensitivity· v130: computational_worldview, anthropic_reasoning_comfort, future_technology_prior, consciousness_intuition. Εμφανίζονται μόνο τα σχετικά slots ανά version. Trust/suspicion ΔΕΝ υπάρχουν εδώ — τα κουβαλάει το core (αποφύγαμε τη διπλομέτρηση).
- **Topic profiles:** έτοιμοι συνδυασμοί των παραπάνω (π.χ. v52: Engineering-anchored / Motive-driven doubter / Anomaly-focused). Manual = τα ορίζεις μόνος.
- **Topic background (μεσαίο βάρος):** occupation, education, training_style, domain_familiarity, topic_interest, prior_exposure — πλαίσιο, όχι άμεσος οδηγός πεποίθησης.
- **Flavor (χαμηλό βάρος):** age, gender, ethnicity, lifestyle, tone — επηρεάζουν διατύπωση, όχι κατεύθυνση.
- **Distribution mode (Mode στο Core box):** `all` = ΚΑΘΕ agent παίρνει αυτό το profile όπως
  το έχεις διαμορφώσει· `equal` = ίση μοιρασιά της βιβλιοθήκης presets (υπόλοιπο round-robin)·
  `random` = κάθε agent τραβάει preset ομοιόμορφα· `more`/`most` = αυτό το profile στο 55%/80%
  και τα υπόλοιπα presets μοιράζονται τα υπόλοιπα. ΠΡΟΣΟΧΗ: random/equal τραβούν από τη
  ΒΙΒΛΙΟΘΗΚΗ presets — οι χειροκίνητες αλλαγές σου στο επιλεγμένο profile ΔΕΝ μπαίνουν στο
  pool (γι' αυτό και το editing κλειδώνει εκεί)· σε more/most μπαίνουν, με το βάρος τους.
- **Topic notes:** ελεύθερο κείμενο, ΔΕΝ ξαναγεμίζει από τα live suggestions (fix 2026-07-17 —
  πριν, το Regenerate έγραφε μέσα του ολόκληρο το suggestion κείμενο). ΠΡΟΣΟΧΗ: αν γραφτεί κάτι
  όσο το topic-background layer είναι ON, μπαίνει στην κάρτα ως «Topic note: …» — άφησέ το κενό
  εκτός αν θες να το διαβάσουν οι agents.
- **Reset κουμπιά (fix 2026-07-17):** το reset του topic-background ξαναρίχνει suggestions ΜΟΝΟ
  για το topic group· του flavor ΜΟΝΟ για το flavor. Πριν, οποιοδήποτε από τα δύο ξαναέριχνε και
  τα δύο (κοινό full-apply μονοπάτι) — αυτό ήταν το «αλλάζουν και τα 2 μαζί».
- **Regenerate suggestions:** νέο τράβηγμα ΚΑΘΕ κλικ (σκόπιμα πιθανοτικό — το RNG του live view
  είναι session-seeded και ανεξάρτητο από το Seed του run· το πραγματικό CSV χρησιμοποιεί το
  seeded RNG του run, άρα η προεπισκόπηση δεν δεσμεύει το πείραμα). Σέβεται τα manual override
  ticks: ό,τι είναι σε manual δεν πατιέται.
- Το generated CSV παίρνει audit στήλες (τι layers ήταν ενεργά κ.λπ.) — ΔΕΝ μπαίνουν στο prompt-facing κείμενο.

## 8. Δίκτυο & πληθυσμός

- **N agents / steps / seed:** μέγεθος, διάρκεια, τυχαιότητα. Ίδιο seed = ίδιο δίκτυο/σειρά αλληλεπιδράσεων.
- **Δίκτυο Barabási–Albert, m_attach=2:** κάθε νέος κόμβος φέρνει 2 ακμές με προτιμησιακή σύνδεση → λίγοι hubs υψηλού βαθμού, συνεκτικό αραιό δίκτυο.
- **Hubs strategy (positive/negative):** τα στοχευμένα beliefs ανατίθενται στους κόμβους με τον ΠΡΑΓΜΑΤΙΚΑ υψηλότερο βαθμό αφού χτιστεί το δίκτυο (όχι κατά σειρά εισόδου). Το hub ΔΕΝ το ξέρει ο ίδιος — καμία persona μεταβλητή «κύρους»· μόνο δομική συχνότητα εμφάνισης ως ομιλητής.
- **Opinion dist (ακριβείς συνταγές):** `uniform` = ίσα πέμπτα στα −2,−1,0,+1,+2 (αν το N δεν
  διαιρείται με το 5, το υπόλοιπο γεμίζει από το −2 προς τα πάνω)· `skewed_positive` = 4/5 των
  agents στο +2 και 1/5 στο −2, ΤΙΠΟΤΑ ενδιάμεσα· `skewed_negative` = καθρέφτης (4/5 στο −2,
  1/5 στο +2)· `positive`/`negative` = ΟΛΟΙ στο +2/−2 (consensus start)· `custom_counts` =
  πέντε αριθμοί με κόμμα, ένας ανά κάδο −2..+2, άθροισμα = Agents. Οι θέσεις ανακατεύονται με
  το Seed: ΠΟΙΟΣ παίρνει ποια τιμή εξαρτάται από το seed, τα πλήθη είναι ακριβή.

## 8β. Reach policies (p_reach) — αλγοριθμική εμβέλεια (ADR-006 Component 3)

Ξεχωρίζει την **εμβέλεια** (ποιος πραγματικά *βλέπει* ένα tweet) από τον **βαθμό** του δικτύου (ποιος ακολουθεί ποιον). Κάθε κατευθυνόμενη ακμή `src → dst` αποκτά ένα `p_reach ∈ [0,1]`: όταν ο `src` μιλήσει, ο `dst` εκτίθεται στο tweet μόνο με πιθανότητα `p_reach` (Bernoulli draw ανά listener event, seeded RNG → reproducible). Υπολογίζεται μία φορά στο edge creation μέσω μιας μόνο function `assign_p_reach(policy, params, src, dst, opinions)` (pluggable pattern: νέα policy = νέα function, όχι refactor). Ζει πάνω στις ADR-004 directed edges.

- **Reach policy (p_reach):** ποια πολιτική αναθέτει το `p_reach`.
  - `uniform` **(default)** — κάθε ακμή παίρνει το ίδιο *p_reach uniform value*. Με τιμή `1.0` είναι **byte-identical** με το προ-ADR-006 baseline. Ερώτημα: *έχει η αραίωση εμβέλειας μόνη της effect;* → dose-response sweep `{1.0, 0.75, 0.5, 0.25, 0.1}`. Feeds: baseline.
  - `homophilic` — `p_reach = sigmoid(k · (1 − 2·norm_dist))`, όπου `norm_dist = |opinion[src] − opinion[dst]| / 4` (4 = μέγιστη απόσταση στην −2..+2 κλίμακα). Όμοιες γνώμες → υψηλό reach, αντίθετες → χαμηλό: μιμείται engagement-based amplification (echo chambers). Feeds: **P30** (Cinelli 2021, μοτίβο Twitter/FB).
  - `shadowban` — τυχαίο υποσύνολο agents (κλάσμα = *shadowban fraction*) παίρνει χαμηλό `p_reach` (= *shadowban value*) σε ΟΛΕΣ τις εξερχόμενες ακμές του· οι υπόλοιποι κρατούν `1.0`. Μιμείται content moderation / περιορισμό διάχυσης. Feeds: **P21** (thesis §5.3 shadow-banning).
- **p_reach uniform value** (default `1.0`): ΜΟΝΟ για `uniform`. Η τιμή εμβέλειας κάθε ακμής.
- **p_reach homophily k** (default `2.0`): ΜΟΝΟ για `homophilic`. Sigmoid sharpness — μεγαλύτερο `k` = πιο απότομη αντίθεση similar/dissimilar.
- **p_reach shadowban fraction** (default `0.1`): ΜΟΝΟ για `shadowban`. Ποσοστό agents που τιμωρούνται.
- **p_reach shadowban value** (default `0.1`): ΜΟΝΟ για `shadowban`. Το `p_reach` των εξερχόμενων ακμών των τιμωρημένων.
- **p_reach enforcement** (default `filter`): ΠΩΣ επιβάλλεται ένα `p_reach < 1.0` (homophilic/shadowban). Δύο διαφορετικά μοντέλα του «περιορισμού»:
  - `filter` **(default, byte-identical)** — ο throttled speaker αφαιρείται από τους υποψήφιους speakers του listener με πιθανότητα `1 − p_reach` (Bernoulli στο **κοινό** RNG) → σπάνια *επιλέγεται*. Μοντελοποιεί «reduced surfacing» (ο αλγόριθμος τον εμφανίζει λιγότερο). Μειονέκτημα: τα RNG draws του κάνουν τις single-seed uniform-vs-treatment συγκρίσεις να **αποκλίνουν** (το reach-effect μπερδεύεται με RNG-path divergence).
  - `suppress` **(fix γ, roads_not_taken 3.11)** — οι υποψήφιοι speakers **δεν** φιλτράρονται, άρα ο speaker επιλέγεται ακριβώς όπως σε ένα matched baseline (ίδια ακολουθία interactions)· αφού επιλεγεί, ένα Bernoulli σε **αφιερωμένο** RNG αποφασίζει αν φτάνει το tweet, και σε αποτυχία το belief-update **δεν εφαρμόζεται**. Μοντελοποιεί «de-ranked / ignored delivery» (το tweet φτάνει στο feed αλλά δεν πείθει). Δίνει **καθαρή single-seed αιτιακή σύγκριση** (το main RNG μένει άθικτο), ~5× φθηνότερη από multi-seed averaging. Feeds: **P21**.

**Ειδικές περιπτώσεις:**
- **Καθαρό P21 πείραμα με `suppress`:** το inactive `uniform + 1.0` ΔΕΝ είναι το σωστό control — η ενεργοποίηση οποιασδήποτε policy φτιάχνει `dir_net` και βάζει `neighbors = dir_net.following`, αλλάζοντας τη σειρά των candidate speakers για κάποιους κόμβους (structural artifact, όχι reach-effect). Το καθαρό control είναι **matched baseline** που περνά κι αυτό από `dir_net`: `shadowban` με `shadowban_value = 1.0` + `enforcement = suppress` (ίδιοι throttled agents μέσω dedicated RNG, reach 1.0 → τίποτα δεν κόβεται) **vs** treatment `shadowban_value = 0.1` + `suppress` → ταυτόσημη ακολουθία interactions (ίδιοι listeners ΚΑΙ speakers), μόνη διαφορά τα suppressed belief-updates. Χρησιμοποίησε `interaction_selection = random` για τέλεια ευθυγράμμιση.
- Κάθε sub-παράμετρος αγνοείται από τις policies που δεν την αφορούν (π.χ. `homophily k` δεν κάνει τίποτα σε `uniform`). Άφησέ τες στα defaults εκτός αν τρέχεις την αντίστοιχη policy. Κενό float στο GUI = ο sim βάζει το δικό του default.
- **Byte-identical εγγύηση:** με `uniform` + `1.0` το φιλτράρισμα εμβέλειας παρακάμπτεται εντελώς (καμία επιπλέον κατανάλωση RNG) → η τροχιά απόψεων είναι ταυτόσημη με προ-ADR-006. Οι υπόλοιπες policies μπαίνουν στη διαδρομή μόνο αν επιλεγούν ρητά.
- **Rewires:** όταν το δίκτυο εξελίσσεται (ADR-004), οι rewires ΔΕΝ επαναϋπολογίζουν `p_reach` προς το παρόν — η αλληλεπίδραση rewire×policy αφήνεται για επόμενο ADR.
- Η επιλεγμένη policy + οι παράμετροι καταγράφονται στο `run_metrics.json` (config self-doc, Task 3.3), και το `p_reach` κάθε ακμής εξάγεται στο edge CSV (Task 3.3).

## 9. Multiple runs & έξοδοι

- **Multiple runs:** off / same seed (γράφει _1,_2,…) / consecutive seeds (seed, seed+1, … και ενημερώνει το out όνομα). Καλύπτει replication ΜΙΑΣ διαμόρφωσης — για πλέγμα συνθηκών βλ. ROADMAP «batch/matrix mode».
- **Out:** το όνομα φακέλου του run. Convention: `seed<seed>_<tags>_<N>_<steps>_<version>_<claimdate>_<dist>`.
- **Τι γράφει ένα run:** `*_network_opinion_change_*.csv` (απόψεις ανά βήμα — η ραχοκοκαλιά), `*_network_interactions_*.csv` (πλήρες log αλληλ

## 10. Συμπεριφορά scroll (fixes 2026-07-17)

- Ροδέλα πάνω από ΚΛΕΙΣΤΟ combobox: κάνει scroll τη σελίδα — ΔΕΝ αλλάζει πια την τιμή (το
  default του ttk άλλαζε τιμές κατά λάθος όσο σκρολάρεις).
- Ροδέλα με ΑΝΟΙΧΤΟ dropdown: σκρολάρει μόνο τη λίστα του dropdown· η σελίδα από κάτω μένει
  ακίνητη (πριν, το popdown «πήγαινε βόλτα» πάνω από τη σελίδα που κουνιόταν).
- Αλλαγή καρτέλας με ανοιχτό dropdown: το dropdown κλείνει αυτόματα (πριν έμενε να αιωρείται
  πάνω από τη νέα καρτέλα).
- Persona card preview: απέκτησε δικό του scrollbar και δικό του wheel scroll (πάνω/κάτω μέσα
  στο κείμενο, χωρίς να σέρνει τη σελίδα).
