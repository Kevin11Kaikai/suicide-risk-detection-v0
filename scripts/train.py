# scripts/train.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the project root directory to Python path
# This allows importing modules from the src directory

import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from src.dataset import create_data_loader
from src.model import build_model, build_optimizer_scheduler, train_model, evaluate_model, save_model
from src.preprocess import engineer_features

def parse_args():
    """
    Parse command line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - model_name: BERT model variant to use
            - epochs: Number of training epochs
            - batch_size: Batch size for training
            - max_length: Maximum sequence length for BERT
            - lr: Learning rate for optimizer
            - device: Device to run training on (cuda/cpu)
            - save_dir: Directory to save the model
    """
    parser = argparse.ArgumentParser(description="Train suicide detection classifier")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                      help="Name of the pre-trained BERT model to use")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=128,
                      help="Maximum sequence length for BERT")
    parser.add_argument("--lr", type=float, default=2e-5,
                      help="Learning rate for optimizer")
    parser.add_argument("--device", type=str, 
                      default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run training on (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="models/",
                      help="Directory to save the trained model")
    return parser.parse_args()

def main():
    """
    Main training function that:
    1. Loads and preprocesses the data
    2. Sets up the model and training components
    3. Trains the model
    4. Evaluates the model
    5. Saves the trained model
    """
    # Parse command line arguments
    args = parse_args()
    print(f"📡 Device: {args.device}")

    # Load and preprocess data
    df = pd.read_csv("data/Suicide_Detection.csv")
    # Apply feature engineering and create binary labels
    df = engineer_features(df)  # feature engineering + label creation

    # Split data into training and validation sets
    # Using 80% for training, 20% for validation
    # Stratify to maintain class distribution
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # Create data loaders for training and validation
    # Training loader with shuffling, validation loader without shuffling
    train_loader = create_data_loader(train_df, tokenizer, args.max_length, args.batch_size)
    val_loader = create_data_loader(val_df, tokenizer, args.max_length, args.batch_size, shuffle=False)

    # Initialize BERT model for binary classification
    model = build_model(args.model_name, num_labels=2, device=args.device)

    # Set up optimizer and learning rate scheduler
    optimizer, scheduler = build_optimizer_scheduler(model, train_loader, args.epochs, args.lr)

    # Training loop
    for epoch in range(args.epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{args.epochs}")
        # Train for one epoch and get average loss
        avg_loss = train_model(model, train_loader, optimizer, scheduler, args.device)
        print(f"📉 Train Loss: {avg_loss:.4f}")

    # Evaluate model on validation set
    print("\n🔍 Evaluating...")
    preds, labels = evaluate_model(model, val_loader, args.device)
    acc = accuracy_score(actuals, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(actuals, predictions, average='weighted')
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("Classification Report:", classification_report(actuals, predictions, target_names=["Non-suicidal", "Suicidal"]))

    # Save the trained model and tokenizer
    save_model(model, tokenizer, args.save_dir)

if __name__ == "__main__":
    main()
