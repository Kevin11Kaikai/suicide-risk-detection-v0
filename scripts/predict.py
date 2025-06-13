# scripts/predict.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the project root directory to Python path
# This allows importing modules from the src directory
import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from src.model import load_model
from src.preprocess import preprocess_text
from src.analyze import analyze_text
import torch.nn.functional as F

def parse_args():
    """
    Parse command line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - model_dir: the directory of the saved pretrained model
            - text: the text to be predicted
            - max_length: Maximum sequence length for BERT
            - device: Device to run training on (cuda/cpu)
    """
    parser = argparse.ArgumentParser(description="Predict suicide risk from a single text input")
    parser.add_argument("--model_dir", type=str, default="models/")
    parser.add_argument("--text", type=str, required=True, help="Input text to analyze")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"📡 Device: {args.device}")

    # Load model and tokenizer
    model, tokenizer_name = load_model(args.model_dir, args.device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model.eval()

    # Preprocess the input text
    clean_text = preprocess_text(args.text)

    # Tokenize
    inputs = tokenizer(
        clean_text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=args.max_length
    )
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        prob_score = probs[0][pred].item()

    label_map = {0: "non-suicide", 1: "suicide"}

    print(f"\n🧠 Model Prediction: {label_map[pred]} ({pred})")
    print(f"📊 Probability: {prob_score * 100:.2f}%")

    # Run rule-based analyzer
    result = analyze_text(args.text)

    print(f"\n📉 Sentiment Score: {result['sentiment_score']}")
    print(f"🧾 Suicide Words Detected: {result['suicide_keywords']}")
    print(f"👤 First-person Pronouns: {result['first_person_count']}")
    print(f"⚠️ Risk Level (Rule-Based): {result['risk_level']}")

if __name__ == "__main__":
    main()
