# COS 760 - Sentiment Analysis for Sesotho and Setswana
**Improving Sentiment Classification for Low-Resource African Languages Using Transfer
Learning and Data Augmentation**

**Group 7:** Litsiba Palesa | Leotlela Kananelo | Moeti Katleho

---

## Summary

This project investigates sentiment analysis for Sesotho and Setswana, two of the low-resource Southern African Bantu languages absent from major NLP benchmarks. A novel and custom Group 7 Setswana corpus is constructed using distant supervision. The four multilingual transformer models (mBERT, AfriBERTa, XLM-RoBERTa, AfroXLMR) are evaluated against a TF-IDF baseline across two experimental runs conducted in this project: Run 1 (no augmentation) and Run 2 (NLLB-200 back-translation and Groq LLaMA-3 paraphrasing). AfriBERTa consistently achieves the best performance on the target languages.

---

## Repository Structure

```
COS760-GROUP7/
├── data/                          # Augmented training datasets
│   ├── st_train_aug.csv           # Sesotho: NLLB-200 back-translated
│   ├── st_train_aug_groq.csv      # Sesotho: NLLB-200 and Groq paraphrased
│   ├── tn_train_aug.csv           # Setswana G7: NLLB-200 back-translated
│   ├── tn_train_aug_groq.csv      # Setswana G7: NLLB-200 and Groq paraphrased
│   └── tn_cl_train_aug_groq.csv   # Setswana Closed: NLLB-200 and Groq paraphrased
├── notebooks/
│   └── cos760_group7.ipynb        # Main experiment notebook (Stages 0–9)
├── results/
│   ├── comparisons/               # F1 comparison and cross-lingual charts
│   ├── confusion_matrices/        # Per-language confusion matrix plots
│   ├── attention_heatmap.png      # Attention visualisation (AfriBERTa on Sesotho)
│   ├── results_summary.csv        # Full results table across all conditions
│   └── shap_text_plot.png         # SHAP token-level attribution plot
└── README.md
```

---

## Pipeline Overview

| Stage | Description |
|-------|-------------|
| 0 | Setup and Installations |
| 1 | Build Group 7 Setswana custom Corpus (Distant Supervision) |
| 2 | Load the AfriSenti Reference Languages and the Sesotho Dataset |
| 2B | Load the Closed Setswana Dataset (dsfsi/setswana-sentiment) |
| 3 | Preprocessing and Train/Val/Test Splits |
| 4 | Baseline Models (TF-IDF and Logistic Regression) |
| 5 | Transfer Learning (mBERT, AfriBERTa, XLM-RoBERTa, AfroXLMR) |
| 6 | Data Augmentation (Back-translation and LLM Paraphrasing) |
| 7 | Evaluation and Visualisation |
| 8 | Explainability (SHAP and Attention) |
| 9 | Results Summary and Export |

> **Runtime:** For a better and faster runtime experience, the `T4 GPU` is utilised for this project/experiment.
>
> **Corpus Strategy:**
> - **Stage 1** builds the Group 7 Setswana corpus using distant supervision (PuoData, Tswana News, and keyword lexicon).
>   This is the Group 7's own dataset contribution and the primary training source for the Setswana experiments.
> - **Stage 2B** loads the closed `dsfsi/setswana-sentiment` dataset provided for the project at hand.
>   This dataset is kept **separate** from the Group 7 corpus and used as a held-out evaluation benchmark to measure
>   how well models trained on the Group 7 corpus generalise to the independently annotated Setswana data.
> - **Stage 2** loads the AfriSenti Dataset (Hausa and Swahili) as reference languages for cross-lingual comparison only.

---

## Stage 1 - Build Group 7 Setswana Corpus (Distant Supervision)

At the beginning of the project implementation, there was no access to the mentioned Closed Setswana dataset, so
Group 7 decided to construct one using distant supervision from two publicly available Setswana text sources:
- **PuoData** (Marivate et al., 2023): which is a formal Setswana text from DSFSI, University of Pretoria, accessible on hugging face
- **Tswana News** (MTEB, 2023): Setswana news headlines, which is also accessible on hugging face

The Sentiment labels in this dataset are assigned automatically using a Setswana keyword lexicon following Mabokela & Schlippe (2022).
The resulting corpus is the **Group 7 contribution for the Sentiment Analysis Project at hand**. The corpus flows into training and is compared against the closed Setswana dataset in Stage 2B.

### Stage 1B - Preprocessing the Group 7 Corpus

The raw Group 7 corpus text is cleaned before sentiment labelling is applied.
In the steps below, the noise common in web-sourced Setswana text is handled:
URLs, user mentions, hashtag symbols, whitespace, and duplicate entries are cleaned.
Entries shorter than 4 words are discarded since they are considered to carry insufficient sentiment signals.

---

## Stage 2 - Load AfriSenti Reference Languages and the Sesotho Dataset

### How AfriSenti is Used in This Project

The AfriSenti benchmark (Muhammad et al., 2023) covers sentiment-annotated tweets across 14 African
languages, **but neither Sesotho nor Setswana are included**, making this project a relevant contribution for the underresourced African languages.

AfriSenti is used in this project in **two specific ways**:

1. **As the source of the Afrocentric pretrained models** fine-tuned in Stage 5.
   AfriBERTa and AfroXLMR were pretrained on corpora that include AfriSenti data.
   Fine-tuning these models on Sesotho and Setswana is the core transfer learning experiment for the proposed project.

2. **As reference languages for cross-lingual performance comparison.**
   Hausa (`hau`) and Swahili (`swa`) are loaded. Both are African languages
   with reasonably large AfriSenti splits compared to Sesotho and Setswana. Running the same models on these languages
   gives us a benchmark to identify and observe how well models perform on better-resourced languages
   African languages versus low-resource targets, Sesotho and Setswana.

**Note: AfriSenti is NOT used for training Sesotho or Setswana models directly in the implementation.**
It instead serves as the reference backbone and the source of pretrained model knowledge.

---

## Stage 2B - Load the Closed Setswana Dataset

The `dsfsi/setswana-sentiment` dataset is the closed Setswana sentiment dataset provided for the purpose of this project.
It is loaded here as a **separate, independent evaluation benchmark**.

**Why kept separate from the Group 7 corpus:**
- The Group 7 distantly-supervised corpus (in Stage 1) was constructed independently and is the training source.
- Merging both datasets would change the structure of the reference benchmark and make the evaluation ambiguous.
- By keeping them separate, an evaluation of how well models trained on the Group 7 corpus transfer to
  independently annotated Setswana data is conducted, making the Setswana corpus contribution a meaningful and honest test of generalisation.

---

## Stage 3 - Preprocessing and Train/Val/Test Splits

All three datasets (Sesotho, Group 7 Setswana, and closed Setswana) pass through the same
cleaning function to ensure consistency across the entire pipeline.
The Group 7 corpus was already cleaned in Stage 1, but the cleaning function is reapplied in this stage
to ensure and guarantee uniform treatment across all the datasets.

**Preprocessing steps applied to all datasets:**
- URLs and web addresses are removed
- User mentions are anonymised to `@user`
- Excess whitespace gets collapsed

**Train/Val/Test split strategy:** Stratified 70/10/20 is applied across all datasets.

---

## Stage 4 - Baseline Models (TF-IDF and Logistic Regression)

Character n-gram TF-IDF (1–3 grams, 50,000 features) with logistic regression. Character n-grams are preferred over word n-grams for Bantu languages because they capture morphological variation without requiring a dedicated tokeniser.

---

## Stage 5 - Transfer Learning (mBERT, AfriBERTa, XLM-RoBERTa, and AfroXLMR)

### Connection to AfriSenti and the Transfer Learning Approach

The four models fine-tuned during this stage were pre-trained on large multilingual corpora, which include African languages, and for AfriBERTa and AfroXLMR, AfriSenti data was part of the pre-training. The main transfer learning experiment suggested in this project is to fine-tune these models on the Sesotho News Headlines dataset and the Group 7 Setswana corpus.

The hypothesis for this experiment is that models that have been exposed to African language text before this experiment will transfer more effectively to the low-resource languages of Sesotho and Setswana than a general multilingual model like mBERT, which has no Africa-focused pretraining.

For initial experiments with the standard learning rate of 2e-5, AfroXLMR collapsed, and the training loss went to zero, and the validation loss went to NaN. That was a very clear sign of gradient instability. Instead of completely removing AfroXLMR from the comparison, the issue was addressed by adding a per-model learning rate setting, decreasing the learning rate for AfroXLMR to **5e-6**, which is a quarter of the learning rate used for the other models. This adjustment made the training more stable and allowed us to successfully include AfroXLMR for all of the experiments. We find the sensitivity of continued-pretraining models on fine-tuning hyperparameters to be a finding in itself, and note it as something to consider while working with similar models in similar low-resource settings.

### Models

| Model | HuggingFace ID | Learning Rate |
|-------|---------------|---------------|
| TF-IDF + Logistic Regression | — (sklearn) | — |
| mBERT | `bert-base-multilingual-cased` | 2e-5 |
| AfriBERTa | `castorini/afriberta_large` | 2e-5 |
| XLM-RoBERTa | `xlm-roberta-base` | 2e-5 |
| AfroXLMR | `Davlan/afro-xlmr-base` | 5e-6 |

**Training configuration:** batch size 16, 5 epochs, fp16 disabled, seed 42, HuggingFace Trainer API, T4 GPU.

---

## Stage 6 - Data Augmentation (Back-translation and LLM Paraphrasing)

Applied to training splits only, validation and test sets are not augmented in this stage.

- **NLLB-200 back-translation** (`facebook/nllb-200-distilled-600M`): training texts translated to English and back to Sesotho and Setswana, effectively doubling the training set size.
- **Groq LLaMA-3 paraphrasing** (`llama-3.3-70b-versatile`): 200 paraphrases generated per target language on top of the back-translated data.

Hausa and Swahili did not undergo augmentation as they serve as reference conditions for the target languages.

| Dataset | Run 1 Train | After NLLB-200 | After NLLB and Groq |
|---------|-------------|----------------|-------------------|
| Sesotho | 1,545 | 3,090 | 3,290 |
| Setswana G7 | 28,197 | 56,394 | 56,594 |
| Setswana Closed | 2,417 | 4,834 | 5,034 |

---

## Stage 7 - Evaluation and Visualisation

Results are reported across all models, languages, and conditions using macro F1 as the primary metric, following the AfriSenti SemEval-2023 evaluation protocol.

### Key Results

| Condition | Best Model | F1 Macro (Run 1) | F1 Macro (Run 2) |
|-----------|-----------|-----------------|-----------------|
| Sesotho | AfriBERTa | 0.655 | 0.618 |
| Setswana G7 | mBERT | 0.978 | 0.967 |
| Setswana Closed | AfriBERTa | 0.553 | 0.530 |
| Hausa (reference) | AfriBERTa | 0.787 | — |
| Swahili (reference) | AfriBERTa | 0.562 | — |

> **Note:** Setswana G7 scores are inflated by distant supervision label artefacts. The Setswana Closed dataset provides a clearer and more realistic performance measure as compared to the custom Group 7 corpus.

### Outputs Generated

| File | Description |
|------|-------------|
| `results_summary.csv` | Full results table across all models and conditions |
| `f1_comparison.png` | Macro F1 bar charts across all models and language conditions |
| `crosslingual_comparison.png` | Cross-lingual F1 comparison |
| `setswana_corpus_comparison.png` | Label distribution comparison for the two Setswana datasets |
| `cross_corpus_comparison.png` | Setswana three-way comparison (G7 / Closed / Cross-corpus) |
| `confusion_matrices_Sesotho.png` | Confusion matrices for Sesotho |
| `confusion_matrices_Setswana_G7.png` | Confusion matrices for Setswana G7 |
| `confusion_matrices_Setswana_Closed.png` | Confusion matrices for Setswana Closed |
| `confusion_matrices_Hausa.png` | Confusion matrices for Hausa |
| `confusion_matrices_Swahili.png` | Confusion matrices for Swahili |
| `shap_text_plot.png` | SHAP token-level attributions for AfriBERTa on Sesotho |
| `attention_heatmap.png` | Attention heatmap for AfriBERTa on Sesotho |

---

## Stage 8 - Explainability (SHAP and Attention Visualisation)

SHAP TextExplainer applied to the best-performing checkpoint (AfriBERTa on Sesotho). Token-level SHAP values reveal which linguistic features drive and influence the sentiment predictions, helping identify systematic misclassification patterns in Sesotho. Attention heatmaps from the final transformer layer visualise where the model attends when classifying the Sesotho sentiment.

---

## Stage 9 - Results Summary and Export

Final results table printed and saved to `results_summary.csv`. All output files were downloaded to the local machine.

---

## Requirements

```
A Google Account Drive with more than 30GB of space for Checkpoints to be saved properly during runs
Google Colab T4 GPU for better and quick execution
transformers datasets evaluate scikit-learn seaborn matplotlib
shap sentencepiece sacremoses emoji accelerate groq
```

Install via:
```bash
pip install transformers datasets evaluate scikit-learn seaborn matplotlib shap sentencepiece sacremoses emoji accelerate groq
```

---

## References

- Mokhosi, R., Shivachi, C., & Sethobane, M. (2024). A Sesotho news headlines dataset for sentiment analysis. *Data in Brief*, 54.
- Muhammad, S. H., et al. (2023). AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages. *arXiv:2302.08956*.
- Ogueji, K., Zhu, Y., & Lin, J. (2021). Small Data? No Problem! *Proceedings of the 1st Workshop on Multilingual Representation Learning*.
- Alabi, J. O., et al. (2022). AfroXLMR: A Self-Supervised Learning Framework for African Languages. *arXiv:2211.08053*.
- Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL-HLT 2019*.
- Costa-jussa, M. R., et al. (2022). No Language Left Behind. *arXiv:2207.04672*.
- Marivate, V., et al. (2020). Investigating an Approach for Low-Resource Language Dataset Creation. *COLING 2020*.
