"""Write the smoke-test reference from a known-good installation.

    python tests/make_reference.py [--checkpoint spotter.safetensors]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_smoke import REFERENCE, run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    result = run(args.checkpoint)
    os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
    np.savez(REFERENCE, **result)
    print("wrote", REFERENCE)
    for key, value in result.items():
        print(key, value)
