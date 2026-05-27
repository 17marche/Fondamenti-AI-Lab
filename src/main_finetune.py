import torch
import torch.nn as nn
import torch.optim as optim
import os

from config import *
from data_prep import get_financial_data_loaders
from model import SentimentLSTM
from engine import train_epoch, evaluate

def main():
    print(f"Fine-Tuning su Financial PhraseBank ({DEVICE})")
    
    train_loader, valid_loader, vocab_size = get_financial_data_loaders(
        financial_path=FINANCIAL_DATA,
        imdb_path=IMDB_CSV,
        max_vocab_size=MAX_VOCAB_SIZE,
        max_len=MAX_SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )

    # Ripristino modello pre-addestrato (IMDB)
    print("\nIstanziazione del modello e caricamento pesi da IMDB")
    
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=1,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        pad_idx=PAD_IDX
    ).to(DEVICE)
    
    # Carichiamo i pesi salvati
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    print("Pesi pre-addestrati caricati con successo!")

    # Sostituzione testa MLP (1 -> 3)
    print("\nSostituzione dell'ultimo strato per la classificazione a 3 classi")
    
    # Ultimo strato (indice 3) all'interno di nn.Sequential
    model.classifier[3] = nn.Linear(in_features=HIDDEN_DIM, out_features=FINANCIAL_CLASSES)
    
    model = model.to(DEVICE)
    
    # Nuova Loss Function.
    # 3 classi -> CrossEntropyLoss.
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Benchmark di base con pesi casuali
    print("\nValutazione Baseline (MLP casuale)")
    
    baseline_loss, baseline_acc = evaluate(model, valid_loader, criterion, DEVICE, is_binary=False)
    
    print(f"\t-> Baseline Validation Loss: {baseline_loss:.3f}")
    print(f"\t-> Baseline Validation Accuracy: {baseline_acc * 100:.2f}%")
    print("-----------------------------------------------------------------")
    
if __name__ == '__main__':
    main()