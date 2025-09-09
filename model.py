import argparse
from pathlib import Path
import pandas as pd

from core.reasoning import Reasoner
from utils import MODELS_DIR

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model for SightIQ.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with features + 'label' column.")
    parser.add_argument("--out", type=str, default=str(MODELS_DIR / "xgb_model.json"), help="Output model path.")
    parser.add_argument("--label_col", type=str, default="label", help="Name of label column.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    reasoner = Reasoner(model_path=Path(args.out))
    reasoner.train(df, label_col=args.label_col)
    print(f"Saved model to {args.out}")

if __name__ == "__main__":
    main()