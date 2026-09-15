import argparse
from sling_sim.visualization import render_saved

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--animation", action="store_true")
    args = parser.parse_args()
    render_saved(args.folder, args.animation)
