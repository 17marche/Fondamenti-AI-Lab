import torch
from datetime import datetime

def calculate_accuracy(preds, y, is_binary=True):
    """
    Calcola l'accuratezza del modello.
    Gestisce sia per la classificazione binaria
    che per quella multiclasse tramite il flag 'is_binary'.
    """
    if is_binary:
        # 'preds' sono logits. 
        # sigmoide per schiacciarli tra 0 e 1
        # Arrotondiamo: >= 0.5 positivo, < 0.5 diventa negativo
        rounded_preds = torch.round(torch.sigmoid(preds))
        
        correct = (rounded_preds == y).float()
        
    else:
        # 'preds' sono matrici [batch_size, 3].
        # argmax per trovare l'indice (0, 1 o 2) con valore più alto
        predicted_classes = preds.argmax(dim=1)
        
        # Confrontiamo le classi previste con le etichette reali
        correct = (predicted_classes == y).float()

    # percentuale di risposte corrette
    acc = correct.sum() / len(correct)
    
    return acc

def train_epoch(model, iterator, optimizer, criterion, device, is_binary=True):
    """
    Esegue un'intera epoca di addestramento.
    calcola la loss, aggiorna i pesi
    e restituisce la loss e l'accuratezza medie dell'epoca.
    """
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for i, batch in enumerate(iterator):
        text, labels = batch
        
        text = text.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(text)
        
        if is_binary:
            # Il modello produce una shape [64, 1]. Le label reali sono di forma [64].
            # .squeeze(1) trasforma [64, 1] in [64].
            predictions = predictions.squeeze(1)
        
        loss = criterion(predictions, labels)
        
        acc = calculate_accuracy(predictions, labels, is_binary)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if (i + 1) % 50 == 0:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] Batch {i + 1}/{len(iterator)} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device, is_binary=True):
    """
    Esegue un'epoca di validazione o test.
    """
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for batch in iterator:
            text, labels = batch
            
            text = text.to(device)
            labels = labels.to(device)
            
            predictions = model(text)
            
            if is_binary:
                predictions = predictions.squeeze(1)
            
            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels, is_binary)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
