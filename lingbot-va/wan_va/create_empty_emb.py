import argparse
import os
import shutil
import sys
import tempfile

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import load_text_encoder, load_tokenizer


def _torch_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


@torch.no_grad()
def save_empty_embedding(
    text_encoder,
    tokenizer,
    output_path,
    device,
    dtype=torch.bfloat16,
    text_len=512,
    force=False,
):
    if os.path.exists(output_path) and not force:
        print(f"empty_emb.pt already exists: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    text_encoder.eval()
    inputs = tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=text_len,
        truncation=True,
    )
    outputs = text_encoder(inputs.input_ids.to(device))
    empty_emb = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    empty_emb = empty_emb.squeeze(0).detach().cpu().to(dtype)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(output_path),
            delete=False,
            suffix=".tmp",
        ) as tmp_file:
            tmp_path = tmp_file.name
        torch.save(empty_emb, tmp_path)
        shutil.move(tmp_path, output_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    print(f"Saved empty prompt embedding to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create LingBot-VA empty prompt embedding.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/lingbot-va-base")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--text_len", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dtype = _torch_dtype(args.dtype)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = args.output_path or os.path.join(args.dataset_path, "empty_emb.pt")

    tokenizer = load_tokenizer(os.path.join(args.ckpt_path, "tokenizer"))
    text_encoder = load_text_encoder(os.path.join(args.ckpt_path, "text_encoder"), dtype, device)
    save_empty_embedding(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        output_path=output_path,
        device=device,
        dtype=dtype,
        text_len=args.text_len,
        force=args.force,
    )


if __name__ == "__main__":
    main()
