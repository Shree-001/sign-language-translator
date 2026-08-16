import os
import json

WLASL_JSON_PATH = os.path.join("data", "wlasl", "WLASL_v0.3.json")
OUTPUT_CONFIG_PATH = os.path.join("data", "selected_words_wlasl100.json")

def scope_wlasl100():
    if not os.path.exists(WLASL_JSON_PATH):
        print(f"Error: {WLASL_JSON_PATH} not found!")
        return

    with open(WLASL_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Top 100 glosses by frequency in official WLASL dataset
    wlasl100_entries = data[:100]
    
    selected_words = {}
    vocab_summary = {}
    total_train = 0
    total_val = 0
    total_test = 0

    print("="*60)
    print("OFFICIAL WLASL100 BENCHMARK VOCABULARY SCOPING (100 WORDS)")
    print("="*60)

    for entry in wlasl100_entries:
        gloss = entry.get("gloss", "").strip().lower()
        instances = entry.get("instances", [])
        selected_words[gloss] = instances

        splits = {"train": 0, "val": 0, "test": 0}
        for inst in instances:
            split = inst.get("split", "train")
            splits[split] = splits.get(split, 0) + 1

        total_train += splits["train"]
        total_val += splits["val"]
        total_test += splits["test"]

        vocab_summary[gloss] = {
            "total_samples": len(instances),
            "splits": splits
        }

    for i, (word, info) in enumerate(vocab_summary.items(), 1):
        s = info["splits"]
        print(f"{i:>3}. {word:<15}: {info['total_samples']:>2} samples (train: {s['train']:>2}, val: {s['val']:>2}, test: {s['test']:>2})")

    print("="*60)
    print(f"TOTAL INSTANCES IN WLASL100: {total_train + total_val + total_test}")
    print(f"  Train: {total_train} | Val: {total_val} | Test: {total_test}")
    print("="*60)

    output_data = {
        "selected_words": list(selected_words.keys()),
        "vocab_summary": vocab_summary,
        "totals": {
            "train": total_train,
            "val": total_val,
            "test": total_test,
            "total": total_train + total_val + total_test
        }
    }
    
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved WLASL100 vocabulary configuration to {OUTPUT_CONFIG_PATH}")

if __name__ == "__main__":
    scope_wlasl100()
