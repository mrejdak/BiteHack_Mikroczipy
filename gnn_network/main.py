import sys
import argparse
from src.solution import run_demo

def main():
    parser = argparse.ArgumentParser(description="GRouting Satellite Simulation")
    parser.add_argument('--run', action='store_true', help='Run the interactive demo')
    args = parser.parse_args()

    if args.run:
        run_demo()
    else:
        print("Usage: python main.py --run")

if __name__ == "__main__":
    main()
