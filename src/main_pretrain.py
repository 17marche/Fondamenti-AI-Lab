import torch
import torch.nn as nn
import torch.optim as optim
import os

from config import (
    DEVICE, IMDB_CSV, MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH, BATCH_SIZE,
    EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX,
    LEARNING_RATE_PRETRAIN, PRETRAIN_EPOCHS, MODEL_SAVE_PATH
)

from data_prep import get_imdb_data_loaders
from model import SentimentLSTM
from engine import train_epoch, evaluate

def main():
    print(f"Inizio Pre-addestramento su IMDB. Device: {DEVICE}")

    # get_imdb_data_loaders restituisce i due loader e 
    # la dimensione effettiva del vocabolario (puo essere < MAX_VOCAB_SIZE)
    train_loader, valid_loader, vocab_size = get_imdb_data_loaders(
        csv_path=IMDB_CSV,
        max_vocab_size=MAX_VOCAB_SIZE,
        max_len=MAX_SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )

    # modello
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=1,          # 1 neurone per la classificazione binaria di IMDB
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        pad_idx=PAD_IDX
    ).to(DEVICE)

    # ottimizzatore e loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_PRETRAIN, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    # addestramento
    best_valid_loss = float('inf')

    for epoch in range(PRETRAIN_EPOCHS):
        
        # Train_epoch ed evaluate restituiranno Loss e Accuracy medie
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, DEVICE)

        print(f'Epoca: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        # Salvataggio del modello migliore
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("\t => Modello salvato!")

if __name__ == '__main__':
    main()