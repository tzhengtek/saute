# 👨‍🍳 SAUTE: Speaker-Aware Utterance Embedding Unit

---

## 🧠 Overview

**SAUTE** (Speaker-Aware Utterance Embedding Unit) is a lightweight transformer-based architecture tailored for **dialog modeling**. It integrates **speaker-sensitive memory** with **linear attention** to model utterances effectively at the EDU (Elementary Discourse Unit) level — while avoiding the high cost of full self-attention.

SAUTE is especially useful for:
- Multi-turn conversations
- Multi-speaker interactions
- Long-range dialog dependencies

---

## 🧱 Model Architecture
![saute-architecture](https://github.com/user-attachments/assets/7f18d5b8-9c6b-4577-b718-206a34d84535)

> 🔍 SAUTE contextualizes each token with speaker-specific memory summaries built from utterance embeddings.

---

## 🚀 Key Features

- **Speaker Memory Construction**: Builds structured memory matrices per speaker from EDU embeddings.
- **Efficient Linear Attention**: Contextualizes each token using memory summaries without quadratic complexity.
- **Pretrained Transformer Integration**: Can be plugged on top of BERT (frozen or fine-tuned).
- **Minimal Overhead**: Adds only ~2M parameters for substantial MLM performance improvements.

---

## 📈 Performance

| Model                      | Avg MLM Acc | Best MLM Acc |
|---------------------------|-------------|--------------|
| BERT-base (frozen)        | 33.45       | 45.89        |
| + 1-layer Transformer     | 68.20       | 76.69        |
| + 2-layer Transformer     | 71.81       | 79.54        |
| **+ SAUTE (Ours)**        | **72.05**   | **80.40%**   |

> Evaluated on the **SODA** validation set using masked language modeling (MLM).

---

## 🛠️ System Components

- **EDU-Level Encoder**: Mean-pooled BERT embeddings per utterance.
- **Speaker Memory**: Summarized with outer-product accumulations.
- **Contextualization**: Injected into token representations via speaker-specific memory.

---

## 📚 Research Paper

The full methodology, experiments, and technical deep dive are available in our paper:
📄 **[SAUTE_Speaker_Aware_Utterance_Embedding_Unit.pdf](https://github.com/user-attachments/files/20640425/SAUTE_Speaker_Aware_Utterance_Embedding_Unit.pdf)**

---

## Authors

- [Justin Duc](https://github.com/just1truc) — Tsinghua University
- [Timothé Zheng](https://github.com/tzhengtek) — Tsinghua University
- [Paul Laban] — Tsinghua University

---
