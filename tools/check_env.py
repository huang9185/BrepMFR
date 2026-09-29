"""Quick sanity check that the BrepMFR environment is usable.

Run inside the conda env:  python tools/check_env.py
"""
import importlib
import sys

EXPECTED = {
    "torch": "1.13.1",
    "dgl": "1.0.0",
    "pytorch_lightning": "1.7.1",
    "numpy": "1.23.5",
}
OPTIONAL = ["fairseq", "torch_geometric", "prefetch_generator", "scipy", "tqdm", "tensorboard"]

ok = True
print(f"python      {sys.version.split()[0]}")
for name, want in EXPECTED.items():
    try:
        mod = importlib.import_module(name)
        have = getattr(mod, "__version__", "?")
        flag = "OK " if have.startswith(want) else "WARN"
        if flag != "OK ":
            ok = False
        print(f"[{flag}] {name:18s} {have}  (expected {want})")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name:18s} import failed: {exc}")

for name in OPTIONAL:
    try:
        importlib.import_module(name)
        print(f"[OK ] {name}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name:18s} import failed: {exc}")

try:
    import torch
    import dgl

    if torch.cuda.is_available():
        print(f"[OK ] CUDA device: {torch.cuda.get_device_name(0)}")
        g = dgl.graph(([0], [1])).to("cuda")
        print(f"[OK ] DGL graph on {g.device}")
    else:
        ok = False
        print("[FAIL] torch.cuda.is_available() is False - check `nvidia-smi` inside WSL")
except Exception as exc:  # noqa: BLE001
    ok = False
    print(f"[FAIL] GPU check: {exc}")

try:
    sys.path.insert(0, ".")
    import data.dataset  # noqa: F401
    import models.brepseg_model  # noqa: F401
    print("[OK ] BrepMFR modules import")
except Exception as exc:  # noqa: BLE001
    ok = False
    print(f"[FAIL] BrepMFR import: {exc}")

print("\nAll good." if ok else "\nSome checks failed - see above.")
sys.exit(0 if ok else 1)
