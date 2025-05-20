import torch
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr

# Détection du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du tokenizer et du modèle fine-tuné
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("/content/results/checkpoint-15958")
model.to(device)
model.eval()

# Fonction de prédiction
def predict_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    toxic_prob = probs[0][1].item()
    label = "🟥 TOXIC" if toxic_prob > 0.5 else "🟩 NON-TOXIC"
    return f"{label} (probabilité : {toxic_prob:.2f})"

# Interface Gradio
interface = gr.Interface(
    fn=predict_toxicity,
    inputs=gr.Textbox(lines=2, placeholder="Entrez votre phrase ici...", label="Commentaire"),
    outputs=gr.Text(label="Résultat"),
    title="Détection de Toxicité",
    allow_flagging="never",
    live=False
)

if __name__ == "__main__":
    interface.launch()
