from xgboost import XGBClassifier

import os
import sys

MODEL_PATH = "./model"


def train(model_name=None): ...


if __name__ == "__main__":
    if len(sys.argv) > 3:
        print("usage: python train.py [model_name]")
        sys.exit(1)

    model_name = sys.argv[-1] if len(sys.argv) == 3 else None

    if model_name and model_name not in os.listdir(MODEL_PATH):
        print(f"{model_name} is not found.")
        print("usage: python train.py [model_name]")
        sys.exit(1)

    train(model_name)
