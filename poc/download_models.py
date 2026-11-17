import json
from pathlib import Path


MODELS = {
    "encoder": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "reranker": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
}


def main():
    from huggingface_hub import snapshot_download

    for name, (model, revision) in MODELS.items():
        path = Path("models") / name
        snapshot_download(model, revision=revision, local_dir=path,
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
                          ignore_patterns=["onnx/*", "openvino/*", "rust_model.ot", "tf_model.h5"])
        (path / "poc-model.json").write_text(json.dumps({"model": model, "revision": revision}))
        print(name, revision)


if __name__ == "__main__":
    main()
