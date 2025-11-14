import torch
from transformers import BertTokenizerFast, EncoderDecoderModel
import os
import re
import unicodedata

# --- 1. Configuration ---

# Le chemin vers le dossier que vous avez téléchargé
# (Puisque le script est au même niveau, "./" est correct)
MODEL_DIR = "./bert-gpt2-finetuned-v2" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ces valeurs doivent correspondre à celles de votre entraînement
MAX_LENGTH_INPUT = 15
MAX_LENGTH_OUTPUT = 17 

# --- 2. Fonctions de normalisation (simple) ---
# Il est bon de garder une normalisation minimale pour l'input
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """Nettoie la phrase entrée par l'utilisateur."""
    s = unicodeToAscii(s.lower().strip())
    # On garde une normalisation simple
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) # Enlève les caractères non-alphabétiques
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# --- 3. Chargement du modèle et tokenizer ---

print(f"Chargement du modèle et du tokenizer depuis {MODEL_DIR}...")
try:
    # Le tokenizer est chargé depuis le dossier
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    # Le modèle est chargé depuis le dossier
    model = EncoderDecoderModel.from_pretrained(MODEL_DIR)
except Exception as e:
    print(f"ERREUR: Modèle non trouvé dans {MODEL_DIR}. {e}")
    print("Assurez-vous que le dossier 'bert-gpt2-finetuned' est au même niveau que ce script.")
    exit()

# Mettre le modèle sur GPU si possible et en mode évaluation
model.to(device)
model.eval()
print("Modèle chargé.")

# --- 4. Boucle de Chat ---
def chat():
    print("="*50)
    print(f"Chatbot (Modèle: {MODEL_DIR}) prêt !")
    print("Tapez 'q' ou 'quit' pour arrêter.")
    print("="*50)
    
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ['q', 'quit']:
                break

            # 1. Nettoyer et tokenizer l'input (pour l'encodeur BERT)
            norm_input = normalizeString(user_input)
            
            # Le tokenizer prépare tout pour BERT
            inputs = tokenizer(
                norm_input, 
                padding="max_length", # On garde les mêmes paramètres que l'entraînement
                truncation=True, 
                max_length=MAX_LENGTH_INPUT, 
                return_tensors="pt"
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device) # Ne pas oublier le masque !

            # 2. Générer une réponse
            # model.generate() utilise le Beam Search par défaut !
            # Nous ajoutons du "sampling" pour de meilleures réponses (celles de votre entraînement)
            # 2. Générer une réponse (BEAM SEARCH PUR, SANS SAMPLING)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_LENGTH_OUTPUT,
                
                # --- On désactive le sampling et on utilise le Beam Search pur ---
                num_beams=5,             # Utilise un Beam Search de 5
                early_stopping=True,     # S'arrête dès que la phrase est finie
                # -----------------------------------------------------------
                
                num_return_sequences=1,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id,
                decoder_start_token_id=model.config.decoder_start_token_id,
            )

            # 3. Décoder la réponse
            # On décode le premier (et unique) output
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print("Bot:", response)

        except KeyboardInterrupt:
            print("\nAu revoir !")
            break

if __name__ == "__main__":
    chat()