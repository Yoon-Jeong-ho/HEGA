# HEGA

This repository provides a simple implementation of the **Hybrid Embedding-to-Generation Architecture (HEGA)** using the Hugging Face `mistralai/Mistral-7B-v0.1` model.

The `HEGAModel` splits the model into two parts:

- **Embedding phase** (`0 ~ l_cut`): used to encode texts and build a FAISS index.
- **Generation phase** (`l_cut+1 ~`): generates responses conditioned on the prompt and retrieved texts.

Users can configure the number of layers (`l_cut`), the number of documents to retrieve (`k`), and whether the embedding part should also be used during generation.

## Example

```bash
pip install -r requirements.txt
python scripts/hega_example.py --prompt "Tell me about transformers" --l_cut 16 --k 3
```

Add `--use_embedding_for_generation` to continue generation from the embedding layers.
