import os
import json

WLASL_JSON_PATH = os.path.join("data", "wlasl", "WLASL_v0.3.json")
OUTPUT_CONFIG_PATH = os.path.join("data", "selected_words.json")

def scope_vocabulary():
    if not os.path.exists(WLASL_JSON_PATH):
        print(f"Error: {WLASL_JSON_PATH} not found!")
        return

    with open(WLASL_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Map gloss -> list of instance objects
    gloss_map = {}
    for entry in data:
        gloss = entry.get("gloss", "").strip().lower()
        instances = entry.get("instances", [])
        # Count available video instances
        gloss_map[gloss] = instances

    print(f"Total glosses in WLASL_v0.3.json: {len(gloss_map)}")

    # Priority 1: Weather & Assistant query words
    p1_targets = ["tell", "me", "weather", "time", "what", "today", "hot", "cold", "rain"]
    # Priority 2: Core conversational words
    p2_targets = ["hello", "yes", "no", "please", "thanks", "thank", "sorry", "help", "name", "more", "stop", "want", "need"]

    selected_words = {}
    missing_targets = []

    print("\n--- checking Priority 1 (Weather & Query Words) ---")
    for word in p1_targets:
        if word in gloss_map:
            count = len(gloss_map[word])
            print(f"  [Found] '{word}': {count} instances")
            if count >= 10:
                selected_words[word] = gloss_map[word]
            else:
                print(f"    -> Warning: '{word}' has only {count} samples (<10).")
        else:
            print(f"  [MISSING] '{word}' NOT in WLASL vocabulary!")
            missing_targets.append(word)

    print("\n--- checking Priority 2 (Conversational Words) ---")
    for word in p2_targets:
        if word in gloss_map and word not in selected_words:
            count = len(gloss_map[word])
            print(f"  [Found] '{word}': {count} instances")
            if count >= 10:
                selected_words[word] = gloss_map[word]

    # Priority 3: Top visually distinct words with sample counts >= 15 up to ~25-30 total target words
    target_count = 30
    if len(selected_words) < target_count:
        # Sort remaining glosses by number of instances descending
        remaining = [(g, len(insts)) for g, insts in gloss_map.items() if g not in selected_words]
        remaining.sort(key=lambda x: x[1], reverse=True)

        print("\n--- Adding Priority 3 (High-sample WLASL words to reach ~25-30 total) ---")
        for g, cnt in remaining:
            if len(selected_words) >= target_count:
                break
            if cnt >= 15:
                print(f"  [Added] '{g}': {cnt} instances")
                selected_words[g] = gloss_map[g]

    print("\n" + "="*50)
    print(f"FINAL SELECTED VOCABULARY ({len(selected_words)} words):")
    print("="*50)
    vocab_summary = {}
    for word, insts in selected_words.items():
        splits = {"train": 0, "val": 0, "test": 0}
        for inst in insts:
            split = inst.get("split", "train")
            splits[split] = splits.get(split, 0) + 1
        vocab_summary[word] = {
            "total_samples": len(insts),
            "splits": splits
        }
        print(f" - {word:<12}: {len(insts):>2} samples (train: {splits['train']}, val: {splits['val']}, test: {splits['test']})")

    # Save mapping info
    output_data = {
        "selected_words": list(selected_words.keys()),
        "vocab_summary": vocab_summary,
        "missing_targets": missing_targets
    }
    
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved vocabulary scoping info to {OUTPUT_CONFIG_PATH}")

if __name__ == "__main__":
    scope_vocabulary()
