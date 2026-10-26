# MedSigLIP-448: reference notes (2026-10-17)

Research for MS1 (ledger). [D] = documented, source given; [I] = inference; NF = not found; [V] = re-checked by us
against the source or our code. Sources: MedGemma tech report arXiv 2507.05201 (§2.2.2, Table 1, §6); MedGemma 1.5
report arXiv 2604.05081; HF card google/medsiglip-448; HAI-DEF model card and terms; GitHub Google-Health/medsiglip
notebooks. The Nature Medicine version (2026-10-06) is paywalled and was not read.

## 1. What the model is

- SoViT-400M/14 vision tower (27 blocks, width 1152, MLP 4304, 16 heads, pre-LN, final post-LN, MAP attention-pool
  head) + a 400M text tower; gelu tanh-approx. [D][V: our conversion, parity 2.6e-6 vs HF pooler_output]
- It is the Gemma 3 SigLIP-400M encoder further trained contrastively on 33M medical image-text pairs (32.6M
  histopathology patches + 635k other), **mixed at 2% weight into the original WebLI data**. [D][V §2.2.2]
- **Trained at 896 px.** The 448 release is the same weights with down-sampled position embeddings ("only
  difference"); it was never trained at 448. [D][V §2.2.2] 448 → 32×32 = 1,024 tokens, no CLS.
- Same weights as MedGemma's vision encoder; MedGemma 1.5 froze it, so there are no newer vision weights. [D]

## 2. What it saw

| Modality | Pairs | Note |
|---|---|---|
| Histopathology patches | 32,550,599 | ~98% of the medical pairs |
| MIMIC-CXR | 231,483 | |
| EyePACS | 199,258 | |
| CT-US1 | 59,979 | 2D CT slices + reports [V] |
| MRI-US1 | 47,622 | 2D MRI slices + reports [V]; anatomy not described |
| Derm (internal + PAD-UFES) | 53,096 | |
| PMC | 41,853 | |
| Knee X-ray (Mendeley) | 1,469 | |

- MRI ≈ 7.5% of the 635k non-pathology pairs, inside a 2% medical share. Knee MRI: **probable, not confirmed**
  (MedGemma 1.5's MRI set, likely the same provider, lists head / abdomen / knee, axial only). [I]
- RGB conversion: CT = three intensity windows of the same slice in R/G/B [D][V]. MRI = grayscale replicated to
  R=G=B with per-volume min-max (documented for MedGemma 1.5, "similar to the SigLip encoder"). [D for 1.5, I for
  MedSigLIP]. **No documented case of adjacent slices packed into channels** (what our d96 pipeline does).

## 3. Expected preprocessing (official)

Scale to [-1, 1] (mean = std = 0.5); bilinear resize (big_vision `tf.image.resize`, no antialias, to reproduce
Google's numbers); zero-pad to square, full field of view; no augmentation in the notebooks. [D]

## 4. Evidence on performance

- All official results are **frozen** (zero-shot / linear probe): CXR zero-shot 13-finding AUC 0.844 (ELIXR 0.824);
  derm LP 0.881; DR LP 0.857; pathology LP 0.878 (Path Foundation 0.897). No official full fine-tune, MRI or 3D result. [D]
- Third party: pelvic MRI series classification, fine-tuned MedSigLIP 95.1% (zero-shot useless, < majority baseline). [D]
- Frozen **global** embeddings (CLS / pooled) of MedSigLIP and peers are near chance for small lesions while patch-token
  probes recover them (arXiv 2606.11606). [D] → explains our early frozen LB 0.80–0.85 without anything special.
- NF: any published evidence that fine-tuned MedSigLIP beats CNNs at a few thousand cases; any Kaggle top solution
  or knee/MSK MRI fine-tune.

## 5. Fine-tuning practice vs MS1

| | Official classification notebook | General ViT-L / So400m practice | MS1 [V: runner] |
|---|---|---|---|
| Head | GAP of post-LN patch tokens + linear | — | MAP head (fine-tuned) per window → our per-finding attention over windows |
| Backbone LR | 5e-5 uniform | PaliGemma 3e-6–3e-5; EVA-02-L 2e-5–7e-5 | 5e-5 top → 2.6e-6 patch embed (layer decay 0.9) |
| Layer decay | none | 0.65–0.85 | 0.9 (milder) |
| Head LR | same | — | 1e-3 |
| Drop path | — | 0–0.2 | 0.1 |
| Weight decay | 0.01 | 0.05 | 0.05 |
| Warmup | 5 steps | 2–10 epochs / % of steps | 1 epoch |
| Epochs | 3 | 20–50 | 25 cosine, 1% floor |
| EMA / averaging | — | EMA +0.1–0.9% | SWA of epochs 21–25 |
| Precision | fp32 | — | bf16 |
| Augmentation | none | weak beats strong (FT-CLIP) | 1e augmentation |

Verdict [I]: MS1 sits in the normal range; nothing documented says it is off.

## 6. Where MS1 departs from what the model was built for

1. **Channels**: adjacent slices in R/G/B vs grayscale-replicated MRI in pretraining; the patch embedding (which would
   have to adapt) gets the lowest LR (~2.6e-6). Untested whether grayscale input or a higher patch-embed LR helps.
2. **Resolution**: native 896; 448 is a position-embedding downsample. 896 = 4× tokens, ~5.6× compute.
3. **Planes**: only axial MRI documented; knee reading relies on sagittal / coronal.
4. **Pooling**: Google's classifier and MedGemma use patch tokens; we use the (fine-tuned) MAP vector per window.
   The small-lesion evidence is about *frozen* pooled embeddings, so it is a weaker concern here.

## 7. Licence (HAI-DEF Terms of Use, 2024-11-15)

- Fine-tuned weights are "Model Derivatives"; outputs (our predictions) are unrestricted. [D]
- Sharing derivatives with anyone (teammates, a public dataset) is Distribution: include the §3.2 use restrictions,
  a copy of the terms, a modification notice and the HAI-DEF notice line. A private dataset used only by us is
  arguably not Distribution. [D/I] Free click-through on HF and Kaggle Models. Competition-rule fit not checked.

## 8. Inference (for later)

- 448 ≈ 1.46× the compute of 384 (729 tokens); 896 ≈ 5.6× of 448. HF supports `interpolate_pos_encoding`. [D/I]
- T4 = Turing: fp16 only (no bf16, no FlashAttention-2) → SDPA memory-efficient kernel; check fp16 parity of the
  bf16-trained weights. SigLIP fp16 ONNX export keeps embeddings / pooling head in fp32 (cos ≥ 0.99999). [D/I]
- ToMe-style merging: ~25% faster on SigLIP for ~1% accuracy; hurts small lesions most. [D/I]
- Measured: MS1 fold-0 scoring ≈ 220 s for 825 studies × 94 windows on one H100 (~0.27 s/study). [V]
