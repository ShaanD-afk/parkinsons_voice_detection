# Task-specific voice measurements in ItalianPVS: an exploratory participant-level study

## Abstract

**Background:** Parkinsonian speech analysis is vulnerable to small samples, repeated recordings, task differences and sample-rate confounding. **Objective:** characterize task coverage and estimate participant-level acoustic group differences across five sustained Italian vowels, while assessing sensitivity and documenting unsupported task measures. **Methods:** the reconciled elderly-HC/PD cohort included 17 HC and 19 PD across rates; the primary native-16-kHz feature cohort included 17 HC and 12 PD. The saved VA1 reference contains 29 participants. Five first vowels were measured with Praat-derived median F0, jitter, shimmer, HNR, formants and spectral descriptors. Twenty prespecified group contrasts used Mann–Whitney tests with Holm correction and participant-bootstrap intervals. An independent robustness analysis evaluated VI1 shimmer using leave-one-participant-out, robust summaries, and percentile/BCa bootstrap intervals. D1/D2 and B1/B2/FB1 were characterized but not treated as validated participant-level outcomes. **Results:** VI1 local shimmer was the only primary contrast below the original 20-test Holm threshold (HC-minus-PD mean difference 0.0444; n=17/12; adjusted p≈0.012); its direction remained positive under individual deletion and robust summaries, but the adjusted trimmed-mean/Yuen result was ≈0.061. VA1 OOF baseline performance was BA .900/AUC .917 on 29 participants, with substantial internal-validation uncertainty. D1/D2 event candidates and connected-speech VAD remained unvalidated. **Conclusion:** a vowel-specific shimmer association merits independent replication, but the study is exploratory and not clinically validated.

## Introduction and research gap

Acoustic outcomes depend on speech task and recording conditions. A credible small-corpus study should preserve participant as the inferential unit, report uncertainty, distinguish sustained phonation from connected speech, and avoid turning recordings into independent sample size. ItalianPVS provides several vowel and speech tasks but has sample-rate imbalance and unresolved protocols for some tasks. This analysis prioritizes task-specific measurement and uncertainty over model complexity.

## Research objectives

1. Summarize participant/task coverage under the existing reconciliation and native-16-kHz restriction.
2. Estimate HC–PD first-vowel acoustic contrasts at participant level.
3. Reassess the VI1 shimmer observation under influence, robust-statistic and bootstrap sensitivity.
4. Characterize second-vowel, D1/D2 and reading signals without assigning unknown protocols or claiming unvalidated measures.

## Dataset and participants

The supplied manifest contains 831 recordings from 65 group-qualified participant folders. The preserved reconciliation yields 567 HC/PD recordings from 36 people (17 HC, 19 PD). The included native-16-kHz subset is 455 recordings from 29 people (17 HC, 12 PD). The first-vowel statistical table contains 145 task-participant observations. The 29-person VA1 reference contains one recording per participant. YHC are not in primary disease-group contrasts. Reconciliation decisions are unchanged.

## Methods

### Audio and measurements

Audio files were analyzed without modification. The primary group comparison used native 16 kHz because extra 44.1 kHz recordings belonged to PD participants only. The vowel feature pipeline used a candidate longest voiced interval at a 75 Hz reference floor, trimmed by 10% at both ends. Praat autocorrelation pitch used 75–500 Hz and 10 ms step; cycle perturbation used the configured local jitter/shimmer settings; HNR used Praat CC; formants used Burg with fixed 5 kHz maximum-formant setting; spectral descriptors used 25 ms Hann windows and 10 ms hop. Formant tracks are candidate values, not expert validated. Full definitions and settings appear in the feature table.

### Statistical analysis

Participant was the unit. Prespecified primary features were median F0, local jitter, local shimmer and mean HNR for VA1/VE1/VI1/VO1/VU1 (20 tests). The historical analysis reported mean differences, Hedges’ g, Cliff’s delta, participant-bootstrap intervals and Mann–Whitney p-values, with Holm correction over the 20 tests. New VI1 sensitivity work directly rechecked the source feature CSV; used 10,000 participant-bootstrap replicates for percentile/BCa intervals; computed median, 20%-trimmed and 10%-winsorized differences; and deleted each participant in turn. The saved VA1 OOF results were not refit. No classifiers were added.

## Results

The only primary-family contrast surviving Holm correction was VI1 local shimmer (17 HC/12 PD, mean difference 0.04439, historical 95% interval [0.01853, 0.07745], Hedges’ g 0.889, Mann–Whitney p 0.000600, Holm p 0.01199). The independent percentile interval was [0.01818, 0.07563] and BCa interval [0.02246, 0.08619]. All single-participant deletion estimates remained positive (0.03335–0.04739). The 20%-trimmed estimate remained positive, but its adjusted Yuen p≈0.061; therefore significance depends on analysis choice even though the direction is stable. VI1 HNR showed an HC-minus-PD difference around −7.18 dB but Holm p≈0.0815. Other primary contrasts did not survive correction. Refer to the 20-contrast table and Figure 4 for all estimates/intervals.

The VA1 baseline retained its previously reported 29-participant OOF results: BA .900, AUC .917, sensitivity .917 and specificity .882. Fixed-prediction participant bootstrap intervals were BA [.775,1.000] and AUC [.789,1.000], but do not include refitting or fold variability and are not external validation.

Second-vowel distributions were shorter than first variants, but codebook wording is blank. D1/D2 plots often showed separated CV-like events, but event peaks are not validated syllable onsets. B1/B2/FB1 activity/pause candidates changed with VAD threshold and remain unreviewed. No syllable count, VOT, transcript timing, or connected-speech group inference is reported.

## Discussion

The VI1 shimmer direction persists across participant deletion and robust summaries and is therefore less likely to be the consequence of one influential observation. Its Holm-adjusted p-value does not persist under a 20%-trimmed Yuen test. The signal may be task-specific, but a small n, repeated analysis choices, age/sex/device differences and the fact that all extra-rate recordings are PD prevent disease-specific or confirmatory interpretation. The current corpora do not allow direct replication of `/i/` shimmer.

### What the Results Actually Tell Us

The most actionable result is a prespecified, sensitivity-tested hypothesis: replicate local shimmer during sustained Italian `/i/` under a controlled, repeated-token protocol. The five vowel contexts do not support a general claim that every vowel captures the same group signal. Vowel context changes the measurements, and there is no validated second-variant reliability estimate. Connected speech could add pause/prosody information, but current threshold-sensitive masks do not yet support it. The study’s present contribution is transparent task-specific coverage and measurement uncertainty, not clinical discrimination.

### Recommended Next Research Experiments

1. Independent, blinded `/i/` shimmer replication with matched age/sex, fixed recording chain and repeated tokens.
2. Complete manual review and quantify inter-rater agreement for D1/D2 onset and connected-speech masks.
3. Recover protocol sheets and exact reading material before implementing task-specific timing or text alignment.
4. Continue only with measures whose segmentation and measurement validity can be demonstrated. Defer complex classifiers until larger independent cohorts exist.

## Limitations

The native-16-kHz cohort is 17 HC/12 PD. This does not establish adequate power for small effects. Recording rate is group-associated; resampled PD-only additions cannot solve it. Sample/device and demographic confounding remain. The stable vowel segment is a candidate algorithm. No inter-rater or biological test-retest reliability has been measured. 159 items await human review. D1/D2 count protocol and second-vowel definitions are unresolved; exact FB1 text is unavailable. The VA1 estimate is internal cross-validation and not external or clinical evidence.

## Conclusion

VI1 shimmer is an exploratory, directionally robust but method-dependent signal. It should guide a focused replication, not a clinical claim. Other tasks should remain descriptive until human validation and protocol documentation support participant-level measurements.

## References

1. Dimauro G, Caivano D, Bevilacqua V, Girardi F, Napoletano V. VoxTester, software for digital evaluation of speech changes in Parkinson disease. IEEE MeMeA, 2016. [doi:10.1109/MeMeA.2016.7533761](https://doi.org/10.1109/MeMeA.2016.7533761).
2. Dimauro G, Di Nicola V, Bevilacqua V, Caivano D, Girardi F. Assessment of Speech Intelligibility in Parkinson’s Disease Using a Speech-To-Text System. IEEE Access, 2017. [doi:10.1109/ACCESS.2017.2762475](https://doi.org/10.1109/ACCESS.2017.2762475).
3. Exploring robust computer-aided diagnosis of Parkinson’s disease based on various voice signals. Frontiers in Physics, 2022. [doi:10.3389/fphy.2022.1048833](https://doi.org/10.3389/fphy.2022.1048833).
4. Boersma P, Weenink D. Praat, a system for doing phonetics by computer. Glot International, 2001;5(9/10):341–345. [Praat](https://www.fon.hum.uva.nl/praat/).
5. Study dataset citation and task source documentation are maintained in the local audit reports and referenced above; see [task-audit report](../..//reports/italian_task_audit/task_audit_report.md).

Figures: `italian_figures/fig1_italian_cohort_flow`, `fig2_italian_task_coverage`, `fig3_italian_acoustic_distributions`, `fig4_italian_vowel_effect_forest`, `fig5_vi1_shimmer_robustness`, and `fig6_italian_vad_threshold_sensitivity`.
