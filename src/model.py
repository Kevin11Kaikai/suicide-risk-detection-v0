# src/model.py

import torch
from transformers import (
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

from tqdm import tqdm
import os

def build_model(model_name='bert-base-uncased', num_labels=2, device='cpu'):
    """
    Build and initialize a BERT model for sequence classification.
    
    This function:
    1. Loads a pre-trained BERT model
    2. Configures it for binary classification
    3. Moves the model to the specified device
    
    Args:
        model_name (str, optional): Name of the pre-trained BERT model. 
            Defaults to 'bert-base-uncased'.
        num_labels (int, optional): Number of classification labels. 
            Defaults to 2 (binary classification).
        device (str, optional): Device to run the model on ('cpu' or 'cuda'). 
            Defaults to 'cpu'.
    
    Returns:
        BertForSequenceClassification: Configured BERT model for sequence classification
    """
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_attentions=False,  # Don't return attention weights
        output_hidden_states=False  # Don't return hidden states
    )
    return model.to(device)

def build_optimizer_scheduler(model, train_loader, epochs, lr=2e-5):
    """
    Build optimizer and learning rate scheduler for model training.
    
    This function:
    1. Creates an AdamW optimizer with specified learning rate
    2. Sets up a linear learning rate scheduler with warmup
    3. Calculates total training steps
    
    Args:
        model (BertForSequenceClassification): The BERT model to optimize
        train_loader (DataLoader): Training data loader
        epochs (int): Number of training epochs
        lr (float, optional): Learning rate. Defaults to 2e-5.
    
    Returns:
        tuple: (optimizer, scheduler) configured for training
    """
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # No warmup steps
        num_training_steps=total_steps
    )
    return optimizer, scheduler

def train_model(model, train_loader, optimizer, scheduler, device):
    """
    Train the BERT model for one epoch.
    
    This function:
    1. Sets model to training mode
    2. Iterates through training batches
    3. Performs forward pass, loss calculation, and backpropagation
    4. Updates model parameters
    5. Tracks and displays training progress
    
    Args:
        model (BertForSequenceClassification): The BERT model to train
        train_loader (DataLoader): Training data loader
        optimizer (AdamW): Optimizer for updating model parameters
        scheduler: Learning rate scheduler
        device (str): Device to run training on ('cpu' or 'cuda')
    
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=True)

    for batch in progress_bar:
        # Move batch data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Backward pass and optimization
        loss = outputs.loss
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    """
    Evaluate the BERT model on test data.
    
    This function:
    1. Sets model to evaluation mode
    2. Makes predictions on test data
    3. Collects predictions and true labels
    
    Args:
        model (BertForSequenceClassification): The trained BERT model
        test_loader (DataLoader): Test data loader
        device (str): Device to run evaluation on ('cpu' or 'cuda')
    
    Returns:
        tuple: (predictions, true_labels) for computing metrics
    """
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():  # Disable gradient calculation
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get predictions
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    return predictions, true_labels

def save_model(model, tokenizer, save_dir='models/'):
    """
    Save the trained model and tokenizer.
    
    This function:
    1. Creates the save directory if it doesn't exist
    2. Saves the model and tokenizer to the specified directory
    
    Args:
        model (BertForSequenceClassification): The trained model to save
        tokenizer: The tokenizer to save
        save_dir (str, optional): Directory to save the model. Defaults to 'models/'.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

def load_model(save_dir='models/', device='cpu'):
    """
    Load a saved model and its tokenizer.
    
    This function:
    1. Loads the saved model from the specified directory
    2. Gets the tokenizer path from the model config
    3. Moves the model to the specified device
    
    Args:
        save_dir (str, optional): Directory containing the saved model. 
            Defaults to 'models/'.
        device (str, optional): Device to load the model on. Defaults to 'cpu'.
    
    Returns:
        tuple: (model, tokenizer) loaded from the save directory
    """
    model = BertForSequenceClassification.from_pretrained(save_dir)
    tokenizer = model.config._name_or_path
    return model.to(device), tokenizer
