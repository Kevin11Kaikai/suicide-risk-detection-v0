# 🧠 Suicide Risk Detection using BERT

This project is a suicide risk classification pipeline using BERT and psychological feature engineering.

## 📦 Project Structure
suicide-risk-detection-pipeline/
├── data/ # Dataset (CSV)
├── models/ # Trained models (saved after training)
├── scripts/ # CLI entry points: train, evaluate, predict
├── src/ # Core logic (preprocessing, dataset, model, etc.)
├── requirements.txt # Dependencies
├── README.md


---

## 🛠️ Environment Setup

- **Python version: 3.9.23**
- **PyTorch: 2.5.0+cu118** (with CUDA 11.8 support)
- **Transformers: 4.41.1**

⚠️ *To ensure full compatibility, use Python 3.9.x and match CUDA version to your system.*

### 🔧 Setup (Linux/macOS/Windows):

```bash
# Create environment
python3.9 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


## 🚀 How to Run (in the project root directory)

### 1.  Train the model:

```bash
python scripts/train.py --epochs 3 --batch_size 32 --device cuda

### 2. Evaluate the model:
python scripts/evaluate.py --model_dir models/

### 3. Predict using the model:
python scripts/predict.py --text "I feel like I want to die tomorrow."

### Features
✅ BERT-based classification (bert-base-uncased)

✅ VADER sentiment score analysis

✅ Suicide keyword and pronoun detection

✅ Risk level scoring (rule-based)

✅ Stratified train/test split

✅ Support for GPU (--device cuda)

✅ CLI-based modular design

## 📊 Dataset

This project uses the **[Suicide Watch Dataset from Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)**, which contains Reddit posts labeled as `suicide` or `non-suicide`.

To use this dataset:

1. Visit the Kaggle page: [https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
2. Download `Suicide_Detection.csv`
3. Place the file into the `data/` folder of this project

> ⚠️ **Note:** The `data/` folder is excluded from version control via `.gitignore`. You must manually download the dataset before training or evaluating.

---

## 🧠 Model Weights

After training the model using `scripts/train.py`, the model artifacts (e.g., `pytorch_model.bin`, `config.json`, `vocab.txt`) will be saved to the `models/` directory.

> ⚠️ This folder is also ignored by Git to prevent uploading large files.  
> You are responsible for keeping your trained model locally or uploading it to external hosting (e.g., HuggingFace, Google Drive).

To load a saved model for evaluation or prediction:

```bash
python scripts/evaluate.py --model_dir models/
python scripts/predict.py --text "I feel completely hopeless"



### Disclaimer
This model is for educational and research purposes only. It is not a substitute for professional mental health assessment or intervention.


---

## 📚 Citation / GitHub

If you find this project useful, please consider giving a ⭐ or citing it:

> Suicide Risk Detection Pipeline (2025).  
> [https://github.com/Kevin11Kaikai/suicide-risk-detection-pipeline](https://github.com/Kevin11Kaikai/suicide-risk-detection-pipeline)

---
