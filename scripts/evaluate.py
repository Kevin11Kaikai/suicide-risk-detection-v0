# scripts/evaluate.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the project root directory to Python path
# This allows importing modules from the src directory


import argparse
import pandas as pd
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.dataset import create_data_loader
from src.model import load_model, evaluate_model
from src.preprocess import engineer_features

def parse_args():
    """
    Parse command line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - model_dir: the directory of the saved pretrained model
            - data_path: the path of the data to be evaluated
            - batch_size: Batch size for evaluation
            - max_length: Maximum sequence length for BERT
            - device: Device to run training on (cuda/cpu)
            - device: Device to run training on (cuda/cpu)
    """
    parser = argparse.ArgumentParser(description="Evaluate trained suicide detection model")
    parser.add_argument("--model_dir", type=str, default="models/")
    parser.add_argument("--data_path", type=str, default="data/Suicide_Detection.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"📡 Device: {args.device}")

    # Load and preprocess data
    df = pd.read_csv(args.data_path)
    df = engineer_features(df)

    # Split (ensure using validation set)
    _, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df = val_df[["processed_text", "label"]].copy()

    # Load model & tokenizer
    model, tokenizer_name = load_model(args.model_dir, args.device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # DataLoader
    val_loader = create_data_loader(val_df, tokenizer, args.max_length, args.batch_size, shuffle=False)

    # Evaluate
    preds, labels = evaluate_model(model, val_loader, args.device)

    # Print report
    print("\n📊 Classification Report:")
    print(classification_report(labels, preds, target_names=["non-suicide", "suicide"]))

    print("\n🧾 Confusion Matrix:")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    main()
