import gzip
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from typing import List, Tuple, Dict

class ZipFitAlignment:
    def __init__(self, compression_threshold: float = 0.7):
        self.compression_threshold = compression_threshold
        self.token_stats = Counter()
        self.cross_entropy_history = []
        
    def compute_compression_ratio(self, text: str) -> float:
        """Compute compression ratio using gzip"""
        text_bytes = text.encode('utf-8')
        compressed = gzip.compress(text_bytes)
        return len(compressed) / len(text_bytes)
    
    def compute_token_redundancy(self, tokens: List[int]) -> float:
        """Compute token redundancy within a sequence"""
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        redundancy = 1.0 - (len(token_counts) / total_tokens)
        return redundancy
    
    def filter_by_compression(self, texts: List[str]) -> List[bool]:
        """Filter texts based on compression ratio threshold"""
        compression_scores = [self.compute_compression_ratio(text) for text in texts]
        return [score <= self.compression_threshold for score in compression_scores]
    
    def track_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Track cross-entropy loss improvement"""
        cross_entropy = nn.CrossEntropyLoss()(logits, labels)
        self.cross_entropy_history.append(cross_entropy.item())
        return cross_entropy.item()
    
    def update_token_stats(self, tokens: List[int]):
        """Update token statistics for redundancy analysis"""
        self.token_stats.update(tokens)

class ZipFitBatchProcessor:
    def __init__(self, 
                 tokenizer,
                 zipfit: ZipFitAlignment,
                 batch_size: int = 32,
                 min_redundancy: float = 0.2):
        self.tokenizer = tokenizer
        self.zipfit = zipfit
        self.batch_size = batch_size
        self.min_redundancy = min_redundancy
        
    def prepare_batch(self, texts: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """Prepare and filter batch based on ZIP-FIT criteria"""
        # Filter by compression ratio
        compression_mask = self.zipfit.filter_by_compression(texts)
        filtered_texts = [text for text, keep in zip(texts, compression_mask) if keep]
        
        if not filtered_texts:
            return None, []
            
        # Tokenize filtered texts
        encodings = self.tokenizer(filtered_texts,
                                 padding=True,
                                 truncation=True,
                                 return_tensors='pt')
        
        # Check token redundancy
        token_lists = encodings['input_ids'].tolist()
        redundancy_scores = [self.zipfit.compute_token_redundancy(tokens) 
                           for tokens in token_lists]
        
        # Filter by redundancy
        redundancy_mask = [score >= self.min_redundancy for score in redundancy_scores]
        final_indices = [i for i, keep in enumerate(redundancy_mask) if keep]
        
        if not final_indices:
            return None, []
            
        # Update token statistics
        for tokens in token_lists:
            self.zipfit.update_token_stats(tokens)
            
        # Prepare final batch
        final_encodings = {
            key: encodings[key][final_indices] 
            for key in encodings.keys()
        }
        
        final_texts = [filtered_texts[i] for i in final_indices]
        return final_encodings, final_texts

class ZipFitTracker:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loss_window = []
        self.improvement_history = []
        
    def update(self, loss: float) -> float:
        """Track and compute improvement in cross-entropy loss"""
        self.loss_window.append(loss)
        if len(self.loss_window) > self.window_size:
            self.loss_window.pop(0)
            
        if len(self.loss_window) >= 2:
            avg_previous = sum(self.loss_window[:-1]) / (len(self.loss_window) - 1)
            current = self.loss_window[-1]
            improvement = (avg_previous - current) / avg_previous
            self.improvement_history.append(improvement)
            return improvement
        return 0.0
    
    def get_average_improvement(self) -> float:
        """Get average improvement over the tracking window"""
        if not self.improvement_history:
            return 0.0
        return sum(self.improvement_history) / len(self.improvement_history)

def train_with_zipfit(model, 
                     tokenizer,
                     train_texts: List[str],
                     learning_rate: float = 2e-5,
                     num_epochs: int = 3):
    """Training loop with ZIP-FIT alignment"""
    
    # Initialize ZIP-FIT components
    zipfit = ZipFitAlignment()
    batch_processor = ZipFitBatchProcessor(tokenizer, zipfit)
    tracker = ZipFitTracker()
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        # Process data in batches
        for i in range(0, len(train_texts), batch_processor.batch_size):
            batch_texts = train_texts[i:i + batch_processor.batch_size]
            
            # Apply ZIP-FIT filtering
            batch_encodings, filtered_texts = batch_processor.prepare_batch(batch_texts)
            if batch_encodings is None:
                continue
                
            # Forward pass
            outputs = model(**batch_encodings)
            loss = outputs.loss
            
            # Track improvement
            improvement = tracker.update(loss.item())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        # Epoch statistics
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_improvement = tracker.get_average_improvement()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Improvement: {avg_improvement:.4f}")
        
    return model, zipfit, tracker