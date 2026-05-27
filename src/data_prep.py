import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
import re

# Importiamo le costanti dal config
from config import (PAD_IDX, UNK_IDX, FINANCIAL_DATA)


def clean_text(text):
    """
    Prende una stringa grezza, la converte in minuscolo e 
    rimuove punteggiatura e tag HTML.
    Restituisce una lista di parole (token).
    """
    
    text = str(text).lower()
    
    # tag HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # punteggiatura e i caratteri speciali
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # gli spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    
    # la stringa pulita -> lista di parole ("the stock went up" -> ['the', 'stock', 'went', 'up'])
    tokens = text.split()
    
    return tokens


def build_vocab(cleaned_texts, max_vocab_size):
    """
    Prende tutti i testi puliti, 
    conta le frequenze e crea il dizionario {parola: indice}.
    """
    
    word_counts = Counter()
    
    # occorrenze di ogni singola parola
    for tokens in cleaned_texts:
        word_counts.update(tokens)
        
    # solo le parole più frequenti e sottraiamo 2 (posti sono riservati ai token speciali)
    most_common_words = word_counts.most_common(max_vocab_size - 2)
    
    # PAD_IDX = 0, UNK_IDX = 1
    word_to_idx = {
        '<PAD>': PAD_IDX,
        '<UNK>': UNK_IDX
    }
    
    # creazione dizionario
    next_idx = max(PAD_IDX, UNK_IDX) + 1
    for idx, (word, count) in enumerate(most_common_words):
        word_to_idx[word] = next_idx
        next_idx += 1

    return word_to_idx


class TextDataset(Dataset):
    """
    Classe PyTorch per gestire i dati. Converte le parole
    in indici e applica il padding/truncation.
    """
    def __init__(self, texts, labels, word_to_idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        label = self.labels[idx]

        # Convertiamo parola -> indice numerico
        indices = []
        for word in tokens:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                # <UNK>
                indices.append(UNK_IDX) 

        # se la sequenza è troppo lunga viene troncata
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        # se la sequenza è più corta viene aggiunto <PAD>
        else:
            padding_length = self.max_len - len(indices)
            indices = indices + [PAD_IDX] * padding_length

        # I testi devono essere interi (LongTensor), il layer di Embedding lavora con indici
        x_tensor = torch.tensor(indices, dtype=torch.long)
        
        # BCEWithLogitsLoss richiede float
        y_tensor = torch.tensor(label, dtype=torch.float32)

        return x_tensor, y_tensor


def get_imdb_data_loaders(csv_path, max_vocab_size, max_len, batch_size):
    """
    Legge i dati, crea il vocabolario congiunto, divide i
    set di IMDB (Train/Validation) e restituisce i DataLoader.
    """
    print("Lettura del dataset IMDB")
    df_imdb = pd.read_csv(csv_path)
    
    # Mappatura e pulizia delle etichette IMDB a numeri (1.0 e 0.0)
    df_imdb['sentiment'] = df_imdb['sentiment'].map({'positive': 1.0, 'negative': 0.0})
    df_imdb=df_imdb.dropna(subset=['sentiment'])
    
    # Pulizia dei testi IMDB
    print("Pulizia dei testi IMDB")
    imdb_texts = [clean_text(text) for text in df_imdb['review'].tolist()]
    imdb_labels = df_imdb['sentiment'].tolist()
    
    # dataset Finanziario (solo per il vocabolario)
    print("Lettura dataset Finanziario solo per il vocabolario")
    financial_texts = []
    with open(FINANCIAL_DATA, 'r', encoding='latin-1') as f:
        for line in f:
            if '@' in line:
                text_part = line.split('@')[0]
                financial_texts.append(clean_text(text_part))
                
    # vocabolario congiunto
    print("Creazione del vocabolario")
    all_texts = imdb_texts + financial_texts
    word_to_idx = build_vocab(all_texts, max_vocab_size)
    vocab_size = len(word_to_idx)
    print(f"Dimensione finale del vocabolario: {vocab_size}")
    
    # dati IMDB Train (80%) e Validation (20%)
    print("Suddivisione dei dati in Train e Validation")
    X_train, X_valid, y_train, y_valid = train_test_split(
        imdb_texts, imdb_labels, test_size=0.2, random_state=42
    )
    
    # Dataset PyTorch
    train_dataset = TextDataset(X_train, y_train, word_to_idx, max_len)
    valid_dataset = TextDataset(X_valid, y_valid, word_to_idx, max_len)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    print("Dati pronti")
    return train_loader, valid_loader, vocab_size

def get_financial_data_loaders(financial_path, imdb_path, max_vocab_size, max_len, batch_size):
    """
    Legge il dataset finanziario, mappa le etichette in 3 classi,ricostruisce
    il vocabolario (identico a quello precedente) e restituisce i DataLoader.
    """
    print("Ricostruzione del vocabolario condiviso")
    
    df_imdb = pd.read_csv(imdb_path)
    imdb_texts = [clean_text(text) for text in df_imdb['review'].tolist()]
    
    fin_texts = []
    fin_labels = []
    
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    with open(financial_path, 'r', encoding='latin-1') as f:
        for line in f:
            if '@' in line:
                text_part, label_part = line.strip().split('@')
                
                fin_texts.append(clean_text(text_part))
                fin_labels.append(label_map[label_part.lower()])
                
    all_texts = imdb_texts + fin_texts
    word_to_idx = build_vocab(all_texts, max_vocab_size)
    
    print("Suddivisione dei dati Finanziari in Train e Validation")
    X_train, X_valid, y_train, y_valid = train_test_split(
        fin_texts, fin_labels, test_size=0.2, random_state=42
    )
    
    # Dataset PyTorch
    train_dataset = TextDataset(X_train, y_train, word_to_idx, max_len)
    valid_dataset = TextDataset(X_valid, y_valid, word_to_idx, max_len)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dati Finanziari pronti! Dataset Train: {len(X_train)} frasi, Valid: {len(X_valid)} frasi.")
    return train_loader, valid_loader, len(word_to_idx)
