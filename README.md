# SAUTEUnit: Speaker-Aware Utterance Transformer Encoder

SAUTEUnit (Speaker-Aware Utterance Transformer Encoder) is a novel architecture designed for dialogue modeling. It combines token-level utterance encoding with dynamic, speaker-specific memory updates, allowing the model to learn both content and speaker interaction patterns efficiently.

---

## Key Features

- **FlashAttention-based Utterance Encoding**: Efficient EDU (Elementary Discourse Unit) encoding with low memory footprint.
- **Speaker Memory Module**: Each speaker has its own evolving hidden state, initialized from their name via a frozen BERT encoder.
- **Cross-Speaker Attention**: Speaker updates incorporate information from other speakers using a learned attention mechanism.
- **Masked Language Modeling (MLM) Objective**: Core training task is to reconstruct masked tokens in utterances.
- **Auxiliary Speaker Prediction Task**: Forces the model to leverage cross-speaker dynamics by predicting the next speaker.
- **Dynamic Batching (Coming Soon)**: Plans to efficiently batch dialogues of similar lengths for high-speed training.

---

## Project Structure

```
sources/
  saute_model.py    # Main SAUTEUnit model components (encoders, speaker module)
  saute_config.py   # Configuration class (inherits from PreTrainedConfig)
  saute_dataset.py  # Dataset class (tokenizes dialogues, prepares inputs)
  saute_trainer.py  # Training setup (coming soon)
README.md           # This file
```

---

## How It Works

At each EDU in a dialogue:
1. The EDU is encoded with a FlashAttention-based Transformer encoder.
2. A mean pooling operation compresses the EDU into a single vector `u_t`.
3. Speaker state `h_s` for the current speaker is updated using `u_t` and cross-speaker context.
4. The model predicts:
   - Masked tokens inside the EDU (MLM)
   - The next speaker ID (Auxiliary Speaker Loss)

This results in both local utterance understanding and long-range dialogue modeling.

---

## Installation

Requires:

- Python 3.10+
- PyTorch 2.1+
- HuggingFace Transformers 4.35+
- FlashAttention 2 (or 3 for Hopper GPUs)

Example:
```bash
pip install torch transformers datasets flash-attn
```

---

## Dataset

Training uses the **MultiDomain-QADialog** dataset, created by [Justin Duc](https://huggingface.co/JustinDuc/MultiDomain-QADialog). 

- It contains dialogue samples segmented into EDUs (Elementary Discourse Units).
- Each EDU is annotated with its corresponding speaker.
- The dataset is multi-domain, covering various dialogue topics, styles, and question-answering formats.

> **Dataset URL**: [Huggingface Datasets - JustinDuc/MultiDomain-QADialog](https://huggingface.co/datasets/JustinDuc/MultiDomain-QADialog)

---

## Architecture

The SAUTEUnit model consists of:

- **FlashTransformerEncoder**: Transformer encoder using FlashAttention layers for fast and efficient EDU encoding.
- **CrossSpeakerEmbeddingsModule**: Module maintaining dynamic per-speaker memory, updated after each EDU using cross-speaker attention.
- **Token Embedding Layer**: Standard word embeddings for token inputs.
- **Positional Embedding Layer**: Added to the EDU tokens to preserve order.
- **Speaker Predictor Head**: Auxiliary head predicting the next speaker to encourage cross-speaker modeling.

Loss Function = MLM Loss + `alpha *` Speaker Prediction Loss.

## Citation

If you use SAUTE or extend it in research, please consider citing the repository and original authors.

---

## License

MIT License.

---

## TODO

- Dynamic batching implementation
- Deepspeed integration
- Fine-tuning scripts
- Full HuggingFace `Trainer` support

