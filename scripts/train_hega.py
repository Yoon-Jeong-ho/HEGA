import argparse
import os
from torch.utils.data import DataLoader
import torch
from hega import HEGAModel


def dummy_dataset(num_samples: int = 1000):
    texts = [f"Sample text {i}" for i in range(num_samples)]
    for t in texts:
        yield t


def collate(batch, tokenizer):
    enc = tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True)
    enc['labels'] = enc['input_ids'].clone()
    return enc


def train_phase(model: HEGAModel, dataloader: DataLoader, epochs: int, lr: float):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.model.parameters()), lr=lr)
    for _ in range(epochs):
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model.model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def main():
    parser = argparse.ArgumentParser(description="Train HEGA")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--l_cut", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--emb_epochs", type=int, default=1)
    parser.add_argument("--gen_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="hega_model")
    parser.add_argument(
        "--gen_from_l_cut",
        action="store_true",
        help="Start generation from L+1 instead of layer 0",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.gpus))

    hega = HEGAModel(
        model_name=args.model,
        l_cut=args.l_cut,
        use_embedding_for_generation=args.gen_from_l_cut,
        n_gpus=args.gpus,
    )

    dataset = list(dummy_dataset())
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=lambda b: collate(b, hega.tokenizer),
    )

    hega.unfreeze_embedding_layers()
    hega.freeze_generation_layers()
    train_phase(hega, loader, args.emb_epochs, lr=1e-5)

    hega.freeze_embedding_layers()
    hega.unfreeze_generation_layers()
    train_phase(hega, loader, args.gen_epochs, lr=1e-5)

    gen_tag = "fromL" if args.gen_from_l_cut else "from0"
    save_name = f"{args.model.replace('/', '_')}_cut{args.l_cut}_{gen_tag}_gpus{args.gpus}"
    output_path = os.path.join(args.output_dir, save_name)
    os.makedirs(output_path, exist_ok=True)
    hega.save(output_path)


if __name__ == "__main__":
    main()
