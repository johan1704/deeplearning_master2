# 1. Import + désactivation W&B
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os

os.environ["WANDB_DISABLED"] = "true"

# 2. Charger les données (réduction à 1000 lignes pour test rapide)
try:
    df = pd.read_csv('train 8.csv')
    print(f"Data loaded successfully. Original shape: {df.shape}")
    df = df.sample(n=1000, random_state=42).reset_index(drop=True)  # 🔁 Réduction
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please check the file path.")
    exit()

# 3. Vérification colonnes
if not all(col in df.columns for col in ['comment_text', 'toxic']):
    print("Error: CSV must contain 'comment_text' and 'toxic' columns")
    exit()

df = df.dropna(subset=['comment_text'])
df['toxic'] = df['toxic'].astype(int)

# 4. Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['comment_text'], df['toxic'], test_size=0.2, random_state=42
)

# 5. Tokenization (max_length réduit à 64)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=64)

# 6. Dataset
class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = ToxicDataset(train_encodings, train_labels)
test_dataset = ToxicDataset(test_encodings, test_labels)

# 7. Modèle
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    ignore_mismatched_sizes=True
)

# 8. Arguments d'entraînement (réduction du nombre d’époques)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,                     
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
    logging_dir='./logs',                   
    fp16=True        
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# 10. Évaluation
results = trainer.evaluate()
print(f"\nEvaluation results:")
print(f"Loss: {results['eval_loss']:.4f}")
print(f"Accuracy: {results.get('eval_accuracy', 'N/A')}")

# 11. Fonction de prédiction
def predict_toxicity(text, threshold=0.5):
    # Détection du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assure-toi que le modèle est sur le device
    model.to(device)
    model.eval()  # mode évaluation

    # Préparation des inputs sur le même device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Prédiction sans calcul de gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Softmax sur les logits
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    toxic_prob = probs[0][1].item()

    return {
        'text': text,
        'toxic_prob': toxic_prob,
        'prediction': int(toxic_prob > threshold),
        'label': 'TOXIC' if toxic_prob > threshold else 'NON-TOXIC'
    }


# Exemple de test
test_comment = "you are a bad person"
prediction = predict_toxicity(test_comment)
print(f"\nPrediction example:")
print(f"Text: {prediction['text']}")
print(f"Toxic probability: {prediction['toxic_prob']:.4f}")
print(f"Prediction: {prediction['label']}")
