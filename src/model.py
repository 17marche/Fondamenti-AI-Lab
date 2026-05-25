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

if __name__ == '__main__':
    print("--- INIZIO TEST DELL'ARCHITETTURA (SentimentLSTM) ---")
    
    # Importiamo le variabili dal config per il test
    from config import *
    
    try:
        print("\n[TEST 1] Generazione del Tensore Fittizio (Dummy Input)...")
        # Creiamo un batch finto: 64 frasi, lunghe 256 parole.
        # I numeri simulano gli indici del vocabolario (da 0 a 19999)
        dummy_batch = torch.randint(low=0, high=MAX_VOCAB_SIZE, size=(BATCH_SIZE, MAX_SEQUENCE_LENGTH)).to(DEVICE)
        print(f" Forma dummy_batch: {dummy_batch.shape} --> (Atteso: [{BATCH_SIZE}, {MAX_SEQUENCE_LENGTH}])")
        
        # --- TEST FASE 1: IMDB ---
        print("\n[TEST 2] Istanziazione Modello Fase 1 (IMDB - 1 Classe)...")
        model_imdb = SentimentLSTM(
            vocab_size=MAX_VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=1,          # 1 output per IMDB
            n_layers=N_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT,
            pad_idx=PAD_IDX
        ).to(DEVICE)
        
        print(" Esecuzione Forward Pass su IMDB...")
        output_imdb = model_imdb(dummy_batch)
        print(f" Forma Output IMDB: {output_imdb.shape} --> (Atteso: [{BATCH_SIZE}, 1])")
        
        # --- TEST FASE 2: FINANZA ---
        print("\n[TEST 3] Istanziazione Modello Fase 2 (Finanza - 3 Classi)...")
        model_finanza = SentimentLSTM(
            vocab_size=MAX_VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=FINANCIAL_CLASSES,  # 3 output per Finanza
            n_layers=N_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT,
            pad_idx=PAD_IDX
        ).to(DEVICE)
        
        print(" Esecuzione Forward Pass su Finanza...")
        output_finanza = model_finanza(dummy_batch)
        print(f" Forma Output Finanza: {output_finanza.shape} --> (Atteso: [{BATCH_SIZE}, 3])")
        
        # --- TEST 4: CONTEGGIO PARAMETRI ---
        print("\n[TEST 4] Analisi della Complessità del Modello...")
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f" Parametri addestrabili totali: {count_parameters(model_imdb):,}")
        
        print("\n--- TUTTI I TEST SUPERATI CON SUCCESSO! L'ARCHITETTURA È PRONTA. ---")
        
    except Exception as e:
        print(f"\n[ERRORE DURANTE IL TEST]: {e}")
