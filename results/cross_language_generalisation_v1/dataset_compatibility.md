# Dataset and task compatibility

## Decision

ItalianPVS VA1 versus Figshare sustained `/a/` is **comparable with explicit limitations**. Both source descriptions identify phonation of /a/: the Italian FILE CODES workbook says “2 phonation of the vocal ‘a’”; the Figshare item describes prolonged /a/ by telephone. The exact Italian duration/repetition protocol is not documented in the files reviewed. Figshare language is not established. Thus the matched phonetic target supports an exploratory cross-dataset test, but it does not establish cross-language generalisation.

## Compatibility matrix

| dataset | language_certainty | task_and_phonation | phonetic_content | sample_rate_and_bandwidth | device_environment_and_covariates | participants_by_class | recordings_per_participant | feature_compatibility | comparison_classification | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ItalianPVS VA1 | Italian confirmed at dataset level | sustained vowel; FILE CODES wording: “2 phonation of the vocal a” | /a/ target; exact duration/repetition unspecified | 16 kHz mono primary cohort | dataset recording chain; demographics available | 17 HC / 12 PD | one included VA1 recording/person | same raw feature code applied here | Comparable with explicit limitations | Eligible for zero-shot cross-dataset; not proof of cross-language transfer |
| Figshare | language not established; do not label English | prolonged /a/ by telephone per source item | /a/ target | 8 kHz mono FLOAT; telephone | telephone recordings; age/sex workbook available | 41 HC / 40 PD | one waveform/person in local copy | same raw feature code applied here; 8 kHz limits bandwidth | Comparable with explicit limitations | Eligible for zero-shot cross-dataset; domain, device, rate and possibly language differ |
| MDVR-KCL ReadText | English source task | connected reading text | lexical text; not sustained /a/ | 44.1 kHz mono PCM24 original; candidate features use 16k copy | reading setup differs; age/sex not in local files | 21 HC / 16 PD | one local read recording/person | candidate VAD/prosody; human mask review absent | Not suitable for direct cross-dataset test against sustained vowels | Within-dataset classifier exploratory only |
| MDVR-KCL SpontaneousDialogue | English source task | spontaneous dialogue; examiner may appear | connected spontaneous speech | 44.1 kHz mono PCM24 original | phone dialogue; speaker attribution unresolved | 21 HC / 15 PD | one local dialogue/person | candidate speech activity/prosody; unvalidated | Not suitable for direct cross-dataset test against sustained vowels | No classifier used: speaker attribution/VAD validity unresolved |
| UCI #174 | language metadata not used for cross-language claim | precomputed sustained voice features | prompt not available in local table | no local raw audio | covariates limited in local feature table | 8 HC / 24 PD | 195 rows / 32 coded participants; repeats collapsed to participant medians | historical MDVP features; not re-extracted | Not suitable for raw-audio external validation | Separate within-dataset exploratory classification |
| UCI #301 train | language not established from local source metadata | precomputed vowels/numbers/sentences/words | task rows grouped per source order | no local raw audio | covariates limited | 20 HC / 20 PD | 26 task rows/person; classifier uses one vowel row/person/task | precomputed extractor; HTN not same as HNR | Not suitable for raw-audio external validation | Separate within-dataset exploratory classification |
| UCI #189 | language not needed for current scope | longitudinal precomputed voice measures | home voice records | no local raw audio | 42 PD; no HC | 0 HC / 42 PD | 5,875 repeated rows; subject-level only | precomputed table | Not suitable for binary group classifier | No HC comparison |

## Why MDVR is excluded from external transfer

MDVR ReadText and SpontaneousDialogue are connected speech, while Italian VA1 and Figshare are sustained phonation. Their speaking task, segmentation assumptions, and feature meanings differ. MDVR reading's candidate VAD features have not been human-reviewed; spontaneous dialogue may contain examiner speech. They can inform a provisional within-dataset exercise, not a direct raw-audio zero-shot test against sustained `/a/`.

## Why UCI is excluded from external raw-audio transfer

UCI #174, #189, and #301 are precomputed feature tables with no local waveforms. Their historical feature extraction settings cannot be rerun under the common raw-audio pipeline. UCI #174 and UCI #301 are analyzed only as separate within-table classifiers. UCI #189 is PD-only and cannot support a binary classifier. The local UCI #174 file resolves to 32 IDs (24 PD/8 HC), unlike the UCI source page narrative (31/23/8); this remains an unresolved source discrepancy.

## Cohort coverage

| Dataset/task | HC | PD | Participant unit |
|---|---:|---:|---|
| ItalianPVS VA1 primary native 16 kHz | 17 | 12 | One VA1 recording/person |
| Figshare prolonged `/a/`, native 8 kHz | 41 | 40 | One local waveform/workbook ID/person |
| MDVR reading text | 21 | 16 | One local reading file/person |
| MDVR spontaneous dialogue | 21 | 15 | One local dialogue file/person |
| UCI #174 | 8 | 24 | 195 rows collapsed by local participant ID to one median row/person |
| UCI #301 training vowels | 20 | 20 | One task row/person/vowel; `/a/`, `/o/`, `/u/` analyzed separately |

No cross-dataset identity linkage is available; IDs are dataset-scoped and pseudonymized. A zero overlap of scoped IDs is a split-integrity check, not proof that no person could appear in two separately sourced cohorts.
