import argparse
import json
from pathlib import Path

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trainspotting-recon",
        description="Reconstruct one object from a set of photos: camera poses, verified depth and object masks.")
    parser.add_argument("--images", type=Path, required=True, help="directory with the photos of the object")
    parser.add_argument("--prompt", required=True, help="short noun phrase naming the object, e.g. 'steam locomotive'")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--weights", type=Path, default=Path.home() / ".cache" / "trainspotting",
                        help="directory for downloaded model weights")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = run(args.images, args.prompt, args.output, args.weights, args.device)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
