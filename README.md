# HEGA

This repository provides a simple implementation of the **Hybrid Embedding-to-Generation Architecture (HEGA)** using the Hugging Face `mistralai/Mistral-7B-v0.1` model.

The `HEGAModel` splits the model into two parts:

- **Embedding phase** (`0 ~ l_cut`): used to encode texts and build a FAISS index.
- **Generation phase** (`l_cut+1 ~`): generates responses conditioned on the prompt and retrieved texts.

Users can configure the number of layers (`l_cut`), the number of documents to retrieve (`k`), and whether generation should start after the embedding part.

## Example

```bash
pip install -r requirements.txt
python scripts/hega_example.py --prompt "Tell me about transformers" --l_cut 16 --k 3
```

Add `--gen_from_l_cut` to continue generation from layer `l_cut+1` instead of starting from layer 0.

### Training with multiple GPUs

`scripts/train_hega.py` demonstrates a two-phase training procedure. Choose between generation from layer 0 or after `l_cut` with `--gen_from_l_cut` and select 1-8 GPUs with `--gpus`.

```bash
python scripts/train_hega.py --gpus 2 --l_cut 16 --emb_epochs 1 --gen_epochs 1 \
    --output_dir models
```

The script first trains layers `0~l_cut` for embedding, then freezes them and
fine-tunes the remaining layers for generation. The resulting model is saved in
`models/` with a name that encodes the chosen settings.
