# 🎙️ SHL Grammar Scoring Engine

**Spoken English audio (45–60 s) → continuous grammar score (0–5)**
Solution to the *SHL Hiring Assessment 2026* Kaggle competition.

| | RMSE | Pearson | MAE |
|---|---|---|---|
| **Train (in-sample)** | **0.1425** | 0.9908 | 0.1067 |
| **Cross-validation** (5-fold, speaker-grouped, mean of 3 fold seeds) | **0.5203** | **0.8584** | 0.4017 |
| **Public leaderboard** (0.6·RMSE + 0.4·(1 − Pearson), lower is better) | **0.3464** for `02_submission.ipynb` · best 0.3429 with the same architecture (top 12 % of 450+) | | |

The train/CV gap is expected for SVR heads and a neural network on 731 clips. Generalisation is measured only on held-out
speakers (speaker-grouped CV, stable to ±0.0002 across fold seeds) and confirmed by the leaderboard, which tracked CV
across every submission.

---

## 1. Pipeline

```
                 ┌──────────────────────────────────────────────┐
  Audio clip ──► │ SNR gate (signal processing)                 │── SNR < 4 dB ──► score 0 (not assessable)
                 └──────────────────────────────────────────────┘
                        │ assessable
        ┌───────────────┼───────────────────────┬──────────────────────────┐
        ▼               ▼                       ▼                          ▼
  Whisper-large-v3   Whisper encoder        HuBERT-large              Hand-built speech
  speech-to-text     embedding              15 s segment embeddings   features (from scratch)
        │            (pretrained DL)        (pretrained DL)
        ▼
  Transcript ──► NLP: Qwen3-4B grammar correction + rubric judge · spaCy syntax · Qwen text embeddings
        │
        ▼
  ML heads: SVR ×4 · Ridge ×2 · LightGBM   +   from-scratch PyTorch fusion network
        │
        ▼
  Non-negative linear stack (trained on speaker-grouped out-of-fold predictions) ──► score 1–5
```

| Stage | What | Why |
|---|---|---|
| **Assessability gate** | Frame-energy SNR (digital silence ignored); SNR < 4 dB → score 0 | Catches all 37 label-0 noise clips; real speech starts at ~10 dB |
| **Speech-to-text** | `faster-whisper` large-v3, fp16, VAD, verbatim filler prompt, `condition_on_previous_text=False`, word timestamps | Grammar must be judged on what was *said*, not on what Whisper "fixes" |
| **Hand-built features** (from scratch, 12) | Speech rate, articulation rate, fillers, repetitions, cut-offs, lexical diversity (MATTR), ASR confidence … — all rates, so 60 s train and 45 s test clips are comparable | Interpretable fluency signals |
| **NLP — LLM grammar correction** | Qwen3-4B-Instruct rewrites the cleaned transcript with minimal edits → edits per 100 words | Direct, explainable error rate (5.4 → 3.6 → 2.4 → 1.1 for bands 2 → 5); corrected text is the candidate feedback |
| **NLP — rubric judge** | Qwen3-4B reads the SHL rubric; score = Σ k·P("k") over digit tokens 1–5 | Continuous, calibrated score |
| **NLP — syntax** | spaCy POS / dependency features (17) | Grammatical complexity |
| **Pretrained representations** | Whisper-large-v3 encoder (layers 28–32), HuBERT-large on 15 s segments (14–22), Qwen3-4B mean-pooled text (12–21) and rubric-prompted last token (23–29) | Fixed embeddings; layers chosen by CV |
| **ML heads** | Scaler → PCA 128 → RBF-SVR per embedding; Ridge; LightGBM on 37 tabular features | Regularised heads suit 731 samples |
| **From-scratch DL** | PyTorch MLP fusing all four embeddings (PCA-32 each, fitted inside folds) + tabular features; dropout, AdamW, 3-seed average | **Best single model** (CV RMSE 0.554) |
| **Stacking** | `LinearRegression(positive=True)` on speaker-grouped out-of-fold predictions | Stable, interpretable weights; no speaker leakage |

**Stack weights:** Whisper encoder 0.27 · Qwen text 0.26 · HuBERT 0.22 · from-scratch net 0.20 · tabular LightGBM 0.13 · Qwen SVRs 0.10.

---

## 2. Results (speaker-grouped CV, mean of 3 fold seeds)

| Model | RMSE | Pearson | LB metric |
|---|---|---|---|
| From-scratch fusion network (PyTorch) | 0.554 | 0.838 | 0.397 |
| Whisper-encoder SVR | 0.595 | 0.810 | 0.433 |
| Qwen3-4B text Ridge | 0.611 | 0.799 | 0.447 |
| Qwen3-4B text SVR | 0.625 | 0.792 | 0.458 |
| HuBERT segment SVR | 0.630 | 0.787 | 0.463 |
| Qwen3-4B prompted SVR | 0.670 | 0.753 | 0.501 |
| Tabular LightGBM | 0.689 | 0.735 | 0.519 |
| Tabular Ridge | 0.721 | 0.704 | 0.551 |
| **Stack (final)** | **0.520** | **0.858** | **0.369** |

**Ablation:** removing all LLM-text heads costs +0.016, all audio heads +0.013, the from-scratch net +0.002 — text and audio are complementary.

## 3. Validation & robustness

* **Speaker leakage:** 385 speakers found by clustering HuBERT layers 3–6; 74 % of clips share a speaker. Plain K-fold reports
  LB metric 0.337, speaker-grouped 0.369 → all model selection used the grouped (pessimistic) scheme.
* **Stability:** 0.3690 / 0.3689 / 0.3686 over three fold seeds.
* **Adversarial validation (train vs test AUC):** LLM grammar features 0.53 (no shift) · tabular 0.68 · duration 0.71 ·
  embeddings 0.78–0.80 (length, recording conditions, topic). The stack mixes shift-free and shift-sensitive signals.
* **Learning curve:** text models and the fusion net still improve with more clips → data-limited.
* **End-to-end test:** `score_audio()` on unseen test audio reproduces the batch predictions within 0.03.

## 4. Interpretability

For any clip the engine returns the transcript with grammar corrections highlighted, the error rate, the rubric judge's
score, and the exact decomposition of the final score into weight × model prediction.

## 5. What did not work (measured, then rejected)

| Idea | Result | Decision |
|---|---|---|
| WavLM-large embeddings | Strong, but CC BY-SA licence | Replaced by HuBERT (Apache-2.0) |
| T5 grammar-correction model | Non-commercial licence | Replaced by Qwen3 (Apache-2.0) |
| LanguageTool error counts | Too few errors detected in speech | Dropped |
| CTC-vs-Whisper "correction gap" | Measured accent, not grammar | Dropped |
| TF-IDF / grammar-skeleton TF-IDF | Weaker than LLM embeddings | Dropped |
| Attention-pooling network over frames | CV RMSE 0.589 | Replaced by fusion MLP |
| Pseudo-labelling test clips | No gain | Dropped |
| Few-shot LLM judge (speaker-safe anchors) | r 0.584 vs 0.574 zero-shot; no stack gain | Redundant |
| LM surprisal + correction gap | r −0.37 alone; ΔCV −0.0001 in stack | Redundant |
| Speaker smoothing of predictions | ΔCV −0.001, MAE worse | Below noise |

## 6. Leaderboard history

| Submission | Public LB | Change |
|---|---|---|
| L1 blend | 0.5252 | Hand-built features + TF-IDF |
| L2 blend | 0.4442 | + LLM grammar features |
| Stacks with WavLM | 0.3644–0.3709 | Pretrained audio embeddings |
| Segment-level stack | 0.3565 | 15 s segments fix the train/test length mismatch |
| Licence-clean stack + Qwen embeddings | 0.3521 | WavLM → HuBERT, Qwen text embeddings |
| Speaker-grouped stack | 0.3432 | Stacker trained on speaker-grouped folds |
| + from-scratch fusion net | **0.3429** | Best public score |
| **`02_submission.ipynb`** (clean rebuild from raw audio) | **0.3464** | Same architecture; the 0.0035 gap matches its CV (+0.004) |

---

## 7. Repository

```
├── notebooks/
│   └── 02_submission.ipynb     # clean end-to-end pipeline + report (raw audio → submission.csv)
├── submission.csv              # predictions produced by 02_submission.ipynb (216 test clips)
├── LICENSE                     # MIT
└── README.md
```

## 8. Reproduce

1. Join the competition on Kaggle and create a notebook with the competition data attached.
2. Settings → Accelerator **GPU T4 ×1**, Internet **On** (model downloads).
3. Upload `notebooks/02_submission.ipynb` and **Run All** (≈ 2.5 h). It writes `/kaggle/working/submission.csv`.

All randomness is seeded (`SEED = 42`). The competition data is **not** included in this repository, as required by the
competition rules; outputs that would display competition transcripts have been removed from the published notebook.

## 9. Compliance & licences

* No external data; models trained only on the 769 provided clips. No hand labelling of test data.
* All code and models are OSI-licensed with commercial use allowed:
  Whisper / faster-whisper / spaCy / LightGBM (MIT) · HuBERT / Qwen3 (Apache-2.0) · PyTorch / scikit-learn (BSD) · librosa (ISC).
* This repository's code is released under the MIT licence.

**Author:** Neeraj Kumar · Kaggle: [neerajbhartiya121](https://www.kaggle.com/neerajbhartiya121)
