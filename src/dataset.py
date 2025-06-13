# src/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class SuicideDetectionDataset(Dataset):
    """
    Custom PyTorch Dataset class for suicide risk detection using BERT.
    
    This class handles the conversion of text data into BERT-compatible format by:
    1. Tokenizing input text using BERT tokenizer
    2. Adding special tokens (CLS, SEP)
    3. Padding/truncating sequences to fixed length
    4. Creating attention masks
    5. Converting labels to tensors
    
    Args:
        texts (pandas.Series): Series containing preprocessed text data
        labels (pandas.Series): Series containing binary labels (0 or 1)
        tokenizer (BertTokenizer): BERT tokenizer instance
        max_length (int, optional): Maximum sequence length. Defaults to 128.
    
    Returns:
        dict: Dictionary containing:
            - input_ids: Token IDs for BERT input
            - attention_mask: Mask for valid tokens
            - label: Binary label tensor
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize the dataset with texts, labels, and tokenizer.
        
        Args:
            texts (pandas.Series): Preprocessed text data
            labels (pandas.Series): Binary labels (0: non-suicide, 1: suicide)
            tokenizer (BertTokenizer): BERT tokenizer instance
            max_length (int, optional): Maximum sequence length. Defaults to 128.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing:
                - input_ids (torch.Tensor): Token IDs for BERT input
                - attention_mask (torch.Tensor): Mask for valid tokens
                - label (torch.Tensor): Binary label tensor
        """
        # Get text and label for the given index
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]

        # Tokenize text and prepare for BERT
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=self.max_length,  # Truncate if longer
            return_token_type_ids=False,  # Not needed for single sequence
            padding='max_length',  # Pad to max_length
            truncation=True,  # Truncate if longer than max_length
            return_attention_mask=True,  # Return attention mask
            return_tensors='pt'  # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),        # shape: [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # shape: [max_length]
            'label': torch.tensor(label, dtype=torch.long)  # shape: []
        }

def create_data_loader(df, tokenizer, max_length=128, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the suicide detection dataset.
    
    This function creates a DataLoader that:
    1. Wraps the dataset in a SuicideDetectionDataset
    2. Configures batch size and shuffling
    3. Prepares data for training/inference
    
    Args:
        df (pandas.DataFrame): DataFrame containing:
            - processed_text: Preprocessed text data
            - label: Binary labels (0 or 1)
        tokenizer (BertTokenizer): BERT tokenizer instance
        max_length (int, optional): Maximum sequence length. Defaults to 128.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    
    Returns:
        DataLoader: PyTorch DataLoader instance configured for the dataset
    """
    # Create dataset instance
    dataset = SuicideDetectionDataset(
        texts=df['processed_text'],
        labels=df['label'],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create and return DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
