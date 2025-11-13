import numpy as np
import random

# Must add allow_pickle=True because scenarios contains dictionary objects
d = np.load("mixed_pattern_training_set_50.npz", allow_pickle=True)
print("Top-level keys:", d.files)

if "scenarios" in d.files:
    scenarios = d["scenarios"]  # 这是一个 object 数组，每个元素是 dict
    print(f"Total scenarios: {len(scenarios)}")

    # 随机取一个
    s = random.choice(scenarios)

    # 打印里面的键
    print("Keys inside one scenario:", s.keys())

    # 打印主要数据内容
    if "decision_grid" in s:
        print("Grid shape:", s["decision_grid"].shape)
    elif "fire_risk" in s:
        print("Fire risk shape:", s["fire_risk"].shape)
        print("Buildings shape:", s["buildings"].shape)

    print("Metadata:", s.get("metadata", {}))
else:
    print("❌ No 'scenarios' key found in the NPZ file.")
