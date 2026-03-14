#!/usr/bin/env python3
"""
OPT-125M Ranking Predictor Model
Shared model definition for training and inference
"""

import torch
import torch.nn as nn
from transformers import OPTModel

class OPTRankingPredictor(nn.Module):
    """OPT-125M with linear head for ranking score prediction"""
    
    def __init__(self, model_name="facebook/opt-125m"):
        super().__init__()
        print(f"Loading {model_name}...")
        self.opt = OPTModel.from_pretrained(model_name)
        
        # Add linear layer to map hidden states to score
        hidden_size = self.opt.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1)
        
        # Initialize score head
        nn.init.xavier_uniform_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            scores: [batch_size] - ranking scores
        """
        # Get OPT hidden states
        outputs = self.opt(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Use last token's hidden state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        last_token_hidden = last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device),
            sequence_lengths
        ]  # [batch, hidden]
        
        # Project to score
        scores = self.score_head(last_token_hidden).squeeze(-1)  # [batch]
        return scores
