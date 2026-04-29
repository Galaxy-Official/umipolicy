import argparse
import os

from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings, save_embeddings


def main():
    parser = argparse.ArgumentParser(description="Precompute T5 text embeddings for Flexiv training.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to embed.")
    parser.add_argument("--data_dir", type=str, required=True, help="Output directory for t5_embeddings.pkl.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ckpt/google-t5/t5-11b",
        help="Path to the downloaded T5-11b model directory.",
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    output_path = os.path.join(args.data_dir, "t5_embeddings.pkl")
    if os.path.exists(output_path):
        print(f"T5 embeddings already exist at {output_path}. Skipping.")
        return

    print("================================================================================")
    print("Generating offline T5 embeddings for Flexiv training")
    print(f"Prompt: {args.prompt}")
    print(f"T5 model: {args.model_name}")
    print("================================================================================")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        embeddings_dict = generate_t5_embeddings([args.prompt], model_name=args.model_name)
        save_embeddings(embeddings_dict, args.data_dir, check_exists=False)
    except Exception as exc:
        print(f"Failed to generate T5 embeddings offline: {exc}")
        print(f"Make sure the T5-11B repo is available locally at {args.model_name}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
