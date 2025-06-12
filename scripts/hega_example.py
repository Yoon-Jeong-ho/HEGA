import argparse
from hega.model import HEGAModel


def main():
    parser = argparse.ArgumentParser(description="HEGA example")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--l_cut", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--gen_from_l_cut",
        action="store_true",
        help="Start generation after the embedding part",
    )
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    hega = HEGAModel(
        model_name=args.model,
        l_cut=args.l_cut,
        k=args.k,
        use_embedding_for_generation=args.gen_from_l_cut,
    )

    # Example index (toy)
    hega.index_texts([
        "This is a sample document about machine learning.",
        "Another document describes the transformers library.",
        "More texts can be added to build a larger retrieval index."
    ])

    output = hega.generate(args.prompt)
    print("\n=== Generated ===\n")
    print(output)


if __name__ == "__main__":
    main()
