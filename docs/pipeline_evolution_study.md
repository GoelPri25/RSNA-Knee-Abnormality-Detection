# Pipeline evolution study (2026-10-20)

**Task.** Look at the v4 l4copy / PUB-KD-R training pipeline from first principles, using only data we already have. Find where it is bottlenecked, and rank ways to evolve it for the last 2.5 days (deadline 2026-10-23 01:59 CEST = 10-22 23:59 UTC).

**Scope.**
- CPU only. No training, submissions or ledger edits.
- Scripts:
  - `scripts/study_pipeline.py` → `data/pipeline_study/study_pipeline.txt`
  - `scripts/study_input_fidelity.py` → `data/pipeline_study/study_input_fidelity.txt`
- Reused, not redone:
  - `pubkd/students_r/findings_by_band_state.txt` (finding × band × branch × report state)
  - `diag_pubkd_r.txt` and `audit_pubkd_r.txt`
  - ledger rows: r50-GL4, L4C-*, SER-0/1, V4-FUSE, BB-CMP, P3v2-S, 4b, SK-0, JF-0, T-mix, PUB-KD(-R), SURPASS-R

**Abbreviations.**
- m: the LB-PRED recommended metric, ½ clear6 + ½ rapc_12. The LB estimate is −0.104 + 1.066·m, and in paired form ΔLB = 1.10·Δm.
- The "teacher" is the Raptor 384+224 blend. It is in-sample on fold 0, so its fold-0 numbers are descriptive only.
- The teacher-reference LB is still pending. The plan below assumes about 0.949, the public number for this lineage.

---

## 1. Bottleneck findings

| # | Finding | Numbers | Source |
|---|---|---|---|
| 1 | **The targets are the lever, not the fit.** With the same architecture, data and schedule, changing only the targets moved the LB from 0.939 (v4) to 0.943 (ra and rb). The student copies its teacher about as closely as the teacher's two members agree with each other. | Rank distance student~teacher 0.0601 vs teacher 384~224 0.0594; v4~teacher 0.0969. Train KL excess 0.012 nats (v4 0.107). Seeds agree with each other at 0.991. | study_input_fidelity, diag_pubkd_r |
| 2 | **The d96 cache's dropped slices do not limit how well the student copies the teacher**, even though the teacher read native series. | Student~teacher distance does not rise with slice skip on any slot. Difference-in-differences vs the teacher's own member noise (high − low skip third): SagF −0.003 [−0.007, +0.001], CorF −0.001, Ax −0.002, SagNF −0.001, CorNF 0.000. | study_input_fidelity |
| 3 | **Under teacher targets the crop branch adds about nothing**, and the students no longer pay v4's self-location penalty. This explains most of ra's apparent "+0.007 fold-0 miss". | fused − global, in m: ra +0.0007, rb +0.0007, rc −0.0005, rd +0.0001 (v4 +0.0022). v4 fused, MS1 crops → self-located crops: m −0.0029. Paired LB-PRED vs v4-self (0.939), using the students' global file: **ra 0.9432 (actual 0.943)**, rb 0.9450 (actual 0.943). The absolute predictor underrates the whole r50-GL4 family by about 0.004 (v4-self residual +0.0044, ra +0.0036). | study_pipeline A, G |
| 4 | **The LB rewards a different label convention, and the teacher carries it.** The teacher compresses severity and ranks silent cells above report-absent ones. | Severity correlation inside label-v2 positives, teacher vs label-v2: ACL 0.24 vs 0.48, MCL 0.30 vs 0.74, Effusion 0.32 vs 0.61. Silent − absent target, within label-v2 negatives: +0.03 to +0.11. The teacher crosses 0.5 on 6–21% of label-v2-confident cells (MCL, Synovitis, Fracture about 20%). Students' rank shifts by report state: mild/uncertain +0.004 to +0.006, mod+ −0.001 to −0.003. Per-finding m gain vs v4: MCL +0.021, Lateral OA +0.007, Effusion +0.006–0.007, Baker's +0.006; PF OA −0.002 (pure). | study_pipeline C, D; findings_by_band_state |
| 5 | **The full 0.958 blend is a materially different target from Raptor alone.** | R6~Raptor Spearman 0.69 (ACL) to 0.92. Full blend vs Raptor blend: mean rank move 0.064, about the same distance as ra vs v4 (0.036–0.085), which bought +0.004 LB. Cells moving more than 0.1: ACL 39%, Baker's 36%, Contusion 32%, Lateral OA 31%, Fracture 31%, MCL 26%. | study_pipeline E (shards 0–1, 2,938 studies) |
| 6 | **Schedule and capacity are not limiting.** | Label-v2 AUC slope over epochs 20→25 is −0.0006 to +0.0011. The train−val gap is 0.017–0.034 for students vs 0.038 for v4. BB-CMP already closed backbones. | study_pipeline F |

**What this means.** The r50 student is a "member-quality" copy of whatever teacher it gets. Its LB therefore tracks the teacher's label convention and quality, not our input, architecture or schedule. The levers left are:
- a better teacher (the full blend);
- a student that starts with the teacher's own knowledge (rap2 init);
- more distillation studies (all data).

---

## 2. Stage-by-stage view

| Stage | What it carries | Bottleneck? | Evidence |
|---|---|---|---|
| Input (d96 cache: 26/22/18/12/18 slices, FS-heavy) | Enough for the teacher's ranking at member-noise level | **No** for KD. Native slices (4b, +0.003 mild band, a positive tie) stay untested under KD but have a low prior now. | Finding 2; SER-0 (non-FS and axial substitutable) |
| Targets | The label convention plus the teacher's knowledge (Raptor saw OAI) | **Yes, the main lever** | Findings 1, 4, 5 |
| Architecture: global MIL + crop branch | The crop branch earned +0.002 with label-v2; with KD it earns about 0 | Crops are now cost, not gain: half the compute, MS1 at train time, a self-location penalty at inference | Finding 3; L4C-B (with label-v2, crops in training helped the global branch by +0.0044; untested under KD) |
| Optimisation (25 epochs, SWA 21–25) | The fit is at the target-entropy floor | No | Findings 1, 6 |
| Fusion and inference | Fixed 50/50 logit mean | Small. PF OA + Fracture global-only gives m +0.0008 to +0.0012 on 5/5 models. A per-finding α chosen on one seed and scored on the other gives +0.0014 / −0.0001 / +0.0011 / +0.0006. All inside LB noise, and A′ (v4 + this patch) scored LB 0.938 = A-np. | study_pipeline B |
| Evaluation (fold 0) | Paired LB-PRED vs v4-self, using the students' **global** file, tracks KD students (ra exact, rb +0.002) | Usable for r50 KD students. Unusable for rap2, which is in-sample. rapc is built from Raptor agreement, so it may flatter Raptor-trained students: rb was predicted +0.0018 over ra but tied on the LB. | Finding 3 |

**Verdict revisits.**
- **P3 v2 (teacher-student blend, LB 0.936) stays closed for own-data teachers but does not carry over to public teachers.** P3 v2's teacher was one ConvNeXt trained on 75% of our labels, so it had no outside knowledge and the same convention as label-v2. Its α=0.5 softening did not help on the LB, and that also counts against "softness alone is the mechanism".
- **The ra-vs-fit residual is no longer evidence that fold 0 cannot judge KD.** It is mostly the crop self-location penalty that v4 pays and ra does not (finding 3).
- **4b (native slices) is not reopened** (finding 2).

---

## 3. Options ranked by expected LB gain per cost

| Rank | Option | Mechanism | Evidence | Cheap diagnostic → decision it changes | Cost | Expected LB | Risk |
|---|---|---|---|---|---|---|---|
| 1 | **FULL-KD r50**: v4 l4copy r50 on the full-blend targets, pure + mix73, s2026 | A better teacher and the LB-rewarded convention | Finding 5 (moves about as far as the ra−v4 change); ra moved +0.004 while copying at member-noise level | Paired LB-PRED vs v4-self on the **global** file → submit if ≥ ra's m − 0.001. The **teacher-reference LB** sets the transfer ratio: (0.943 − 0.939) / (teacher − 0.939). | 1 node, about 2 h (ra–rd took 105 min for 4 lanes); needs shard 2 | 0.945–0.948, if 40% of the teacher gap transfers | Shard 2 is late; the blend's 0.958 is partly ensemble diversity that a student cannot learn |
| 2 | **rap2 init** (Raptor CoAtNet weights + crop copy), first on Raptor targets (SURPASS-R step 2), then on full targets | The student starts at teacher-member quality; this is born-again self-distillation | Diagnosis: the student copies at member level, so the start point is the ceiling. No direct rap2 evidence yet. | **LB only**, because rap2 is in-sample on fold 0. rap2-Raptor ≥ teacher-reference − 0.002 → run rap2 on full targets; worse → drop rap2 | 1 node (6 epochs, 1×8 studies); 1–2 submissions per variant | Up to teacher level (about 0.949+) on Raptor targets; highest ceiling for the ≥0.958 goal on full targets | No fold-0 gate; 1–2 submissions per variant; slower inference (CoAtNet 384) |
| 3 | **All-data final student** (folds 0–5; teacher targets exist for every study) | 23% more distillation studies (3,524 → 4,349) | Finding 1: the student is limited by the targets it sees, not by fit; all-data gate waived 10-19 | None on fold 0 → judged on the LB only, as the last submission of the winning recipe | 1 node, about 2.5 h | +0.001 to +0.003 | No validation; one LB read |
| 4 | **KD global-only student** (drop the crop branch) as a lane next to FULL-KD | Crops add about 0 under KD; removes MS1 at train time and halves crop compute | Finding 3; L4C-B is the counter-risk (crops in training helped the global branch with label-v2) | Fold-0 m of global-only vs ra's global file: ≥ −0.001 → the final all-data model drops crops (faster inference, which the user wants); worse → keep l4copy | 1 lane (free on a 4-GPU node) | About 0 in score; speed and simplicity gain | Global branch weakens without crop co-training |
| 5 | **Fusion patch**: PF OA + Fracture global-only (fixed prior, nothing fitted) | The crop hurts those two findings in every model | +0.0008 to +0.0012 m on 5/5 models | None needed; bundle it into the next KD submission kernel, never spend a submission on it alone | 0 | +0.001 (noise) | None (A′ showed 0.000 on the LB) |
| 6 | SURPASS-R controls (softened label-v2; teacher order + sharp values) | Separate softness from ordering | D already shows the teacher's *order* differs in the LB-rewarded way; P3 v2 counts against softness alone | Answers "why", but does not give a deployable model | 2 lanes | 0 (mechanism only) | Uses lanes and submissions in the final 2.5 days |

**Not proposed** (closed, and this study adds no new evidence):
- backbones (BB-CMP; capacity not limiting, finding 6);
- schedule, EMA, SWA variants;
- FS/non-FS fusion (V4-FUSE, MC-PAIR);
- per-finding specialists; joint-findings combiner (JF-0);
- seed ensembles (ra~rc 0.991; mean(ra, rc) m +0.0005);
- native slices (finding 2);
- online teacher: student copying is already at the noise floor, so augmentation-consistent teacher views have no deficit to fix and would cost a Raptor forward pass per step.

---

## 4. 2.5-day plan (CEST; deadline Thu 10-23 01:59)

| When | Action | Node-h | Submissions | Depends on |
|---|---|---|---|---|
| Tue 10-20, morning | Read the teacher-reference LB, which sets the transfer ratio and the SURPASS bar. Pre-register FULL-KD (options 1 and 4) in the ledger. | 0 | 0 | — |
| Tue 10-20, after user go | **Node A = SURPASS-R step 2, reshaped:** rap2 pure s2026, rap2 mix73 s2026, **r50 KD global-only (Raptor pure)**, plus one of: rap2 pure s2027 or the user's chosen control. Lane 3 replaces the two mechanism controls (my view, stated once; the user decides). | about 3–5 | 0–2 tonight (rap2 best) | user go |
| Wed 10-21, when shard 2 lands | Build full targets on CPU (`build_pubkd_targets.py`, minutes). **Node B = FULL-KD:** r50 l4copy pure, r50 l4copy mix73, r50 global-only (if lane A3 held up) or a second seed, and rap2 on full targets (pure or mix, whichever won on A). | about 3–5 | 2–3: best r50 FULL-KD (paired LB-PRED on the global file) and rap2-full, both with the PF OA/Fracture patch | shard 2 |
| Wed 10-21, fallback if shard 2 is not in by 12:00 | Full targets on shards 0–1 (2/3 of studies), Raptor rank targets on shard 2 | same | same | — |
| Thu 10-22, morning | **Node C = all-data** (folds 0–5) version of the LB winner (r50 or rap2), 1–2 lanes | about 3 | — | Node B LB |
| Thu 10-22, afternoon/evening | Submit all-data. Keep 2 submissions for final selection: best single plus a lean E3-style pair if it helps. | 0 | 2–3 | — |

**Totals.**
- About 10–13 node-hours of the roughly 100 left.
- About 9–11 of the 12 remaining submissions (2 today, 5 + 5).
- The binding limits are wall time, shard 2 and the user's go, not node-hours.
