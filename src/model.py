import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    """
    Architettura rete Bi-LSTM con testa MLP.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, bidirectional, dropout, pad_idx):
        
        super(SentimentLSTM, self).__init__()
        
        # alcune dimensioni utili
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        # Encoder (Estrazione Feature)
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=pad_idx
        )
        
        # Strato Recorrente (Bi-LSTM)
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
        Forward Pass attraverso la rete.
        input shape: [batch_size, seq_len]
        """
        
        embedded = self.embedding(text)
        
        # LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenazione di Forward + Backward
        if self.bidirectional:
            hidden_final = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden_final = hidden[-1, :, :]
            
        # MLP
        logits = self.classifier(hidden_final)
        
        return logits
