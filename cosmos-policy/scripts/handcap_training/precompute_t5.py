import argparse
import os

from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings, save_embeddings

def main():
    parser = argparse.ArgumentParser(description="Precompute T5 text embeddings offline.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to embed (e.g., 'manipulate object').")
    parser.add_argument("--data_dir", type=str, required=True, help="Output directory to save the t5_embeddings.pkl file.")
    parser.add_argument("--model_name", type=str, default="ckpt/google-t5/t5-11b", help="Path to the downloaded T5-11b model directory.")
    args = parser.parse_args()

    # Create the data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    output_path = os.path.join(args.data_dir, "t5_embeddings.pkl")

    if os.path.exists(output_path):
        print(f"✅ T5 embeddings already exist at {output_path}. Skipping T5 generation.")
        return

    print("================================================================================")
    print(f"🚀 Generating Offline T5 Embeddings for your dataset")
    print(f"🗣️ Prompt: '{args.prompt}'")
    print(f"📥 Loading T5 model from: {args.model_name}")
    print("================================================================================")

    # We patch environment variables to ensure HF Transformers doesn't hang looking for online repos
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Restrict to rank 0 GPU for this script if possible

    try:
        embeddings_dict = generate_t5_embeddings([args.prompt], model_name=args.model_name)
        save_embeddings(embeddings_dict, args.data_dir, check_exists=False)
        print("✅ Offline T5 embeddings generation completed successfully!")
    except Exception as e:
        print(f"❌ Failed to generate T5 embeddings offline: {e}")
        print(f"Please ensure you have downloaded the HuggingFace T5-11b repo locally to {args.model_name}")
        exit(1)

if __name__ == "__main__":
    main()
