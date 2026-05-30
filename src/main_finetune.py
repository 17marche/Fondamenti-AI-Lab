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
    
    # Pesi più bilanciati (Square Root Balancing) per non penalizzare troppo la classe Neutral
    # Classi: 0 (Neg), 1 (Neu), 2 (Pos)
    weights = torch.tensor([2.8, 1.0, 1.8], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights).to(DEVICE)

    # Benchmark di base con pesi casuali
    print("\nValutazione Baseline (MLP casuale)")
    
    baseline_loss, baseline_acc = evaluate(model, valid_loader, criterion, DEVICE, is_binary=False)
    
    print(f"\t-> Baseline Validation Loss: {baseline_loss:.3f}")
    print(f"\t-> Baseline Validation Accuracy: {baseline_acc * 100:.2f}%")
    print("-----------------------------------------------------------------")

    # Congelamento encoder
    print("Congelamento dei pesi di Embedding e Bi-LSTM")
    
    for param in model.parameters():
        param.requires_grad = False
        
    # Scongeliamo TUTTO il blocco Classifier (MLP)
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri addestrabili attivi: {trainable_params:,} (Intero blocco MLP)")

    # Warm-up più rapido con AdamW
    optimizer_warmup = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-3, weight_decay=1e-2)
    
    WARMUP_EPOCHS = 10
    best_warmup_loss = float('inf')
    
    for epoch in range(WARMUP_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer_warmup, criterion, DEVICE, is_binary=False)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, DEVICE, is_binary=False)
        
        print(f"Warm-up Epoca {epoch+1:02} | Train Acc: {train_acc*100:.2f}% | Val Acc: {valid_acc*100:.2f}%")
        
        if valid_loss < best_warmup_loss:
            best_warmup_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH.replace('.pth', '_finetuned.pth'))

    # Benchmark post MLp
    print("\nValutazione Post Warm-up")
    print(f"\t-> Validation Loss: {best_warmup_loss:.3f}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH.replace('.pth', '_finetuned.pth'), map_location=DEVICE))
    _, warmup_acc = evaluate(model, valid_loader, criterion, DEVICE, is_binary=False)
    print(f"\t-> Validation Accuracy: {warmup_acc * 100:.2f}%")
    print("-----------------------------------------------------------------")

    # Full fine tuning
    print("\nFULL FINE-TUNING")
    print("Scongelamento Selettivo (LSTM + MLP, Embedding rimane congelato)")
    
    for param in model.lstm.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    trainable_params_full = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri addestrabili attivi: {trainable_params_full:,} (Rete Completa)")

    # Learning Rate Differenziali (Best Setup: LSTM 1e-4 per performance, Classifier 1e-3)
    optimizer_full = optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, model.lstm.parameters()), 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': filter(lambda p: p.requires_grad, model.classifier.parameters()), 'lr': 1e-3, 'weight_decay': 1e-2}
    ])
    
    FINETUNE_EPOCHS = 30
    best_final_loss = float('inf')
    PATIENCE = 7
    patience_counter = 0
    
    for epoch in range(FINETUNE_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer_full, criterion, DEVICE, is_binary=False)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, DEVICE, is_binary=False)
        
        print(f"Full Fine-Tuning Epoca {epoch+1:02} | Train Acc: {train_acc*100:.2f}% | Val Acc: {valid_acc*100:.2f}%")
        
        if valid_loss < best_final_loss:
            best_final_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH.replace('.pth', '_final_best.pth'))
            patience_counter = 0
            print(f"\t => Modello salvato (Loss: {valid_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[Early Stopping] L'addestramento è stato interrotto all'epoca {epoch+1}")
                break

    # Benchmark finale
    print("\nValutazione Finale")
    print(f"\t-> Final Validation Loss: {best_final_loss:.3f}")
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH.replace('.pth', '_final_best.pth'), map_location=DEVICE))
    _, final_acc = evaluate(model, valid_loader, criterion, DEVICE, is_binary=False)
    
    print(f"\t-> Final Validation Accuracy: {final_acc * 100:.2f}%")
    print("-----------------------------------------------------------------")
    
if __name__ == '__main__':
    main()