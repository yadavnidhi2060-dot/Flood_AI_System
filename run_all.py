"""One-command pipeline:  python run_all.py   (generate sample data if no data -> clean -> train -> figures)"""
from src import config, eda, preprocessing, train
from src.generate_sample_data import main as make_sample
import sys


def main() -> None:
    if not config.OBSERVED_FILE.exists() and not config.SAMPLE_FILE.exists():
        print("== Step 0: generating SYNTHETIC sample data ==")
        sys.argv = [sys.argv[0]]
        make_sample()
    print("\n== Step 1: cleaning & feature engineering ==")
    preprocessing.run()
    print("\n== Step 2: training & evaluation ==")
    train.run()
    print("\n== Step 3: EDA figures ==")
    eda.save_all()
    print("\nDone. Start the dashboard with:  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
