import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    """
    Architettura rete Bi-LSTM con testa MLP.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, bidirectional, dropout, pad_idx):
        
        super(SentimentLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx
        
        # Encoder (Estrazione Feature)
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=pad_idx
        )
        
        # Strato Ricorrente (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            dropout=dropout,
            batch_first=True
        )

        # Classifier (MLP)
        
        # Bi-LSTM -> concateniamo -> doppio dell'hidden_dim.
        mlp_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=mlp_input_dim, out_features=hidden_dim),
            
            nn.ReLU(),
            
            # previene overfitting
            nn.Dropout(p=dropout),
            
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )

    def forward(self, text):
        """
        Forward Pass attraverso la rete con Masked Mean Pooling.
        input shape: [batch_size, seq_len]
        """
        # Creazione maschera [batch_size, seq_len, 1] per ignorare il padding
        mask = (text != self.pad_idx).unsqueeze(-1).float()
        
        embedded = self.embedding(text)
        
        # LSTM output: [batch_size, seq_len, hidden_dim * num_directions]
        output, (hidden, cell) = self.lstm(embedded)
        
        # Applichiamo la maschera all'output
        masked_output = output * mask
        
        # Mean Pooling: media degli stati solo sui token reali
        sum_masked = torch.sum(masked_output, dim=1)
        num_tokens = torch.sum(mask, dim=1).clamp(min=1e-9)
        
        pooled_output = sum_masked / num_tokens
            
        # MLP
        logits = self.classifier(pooled_output)
        
        return logits
