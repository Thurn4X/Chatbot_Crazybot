import os
import json
import numpy as np
import faiss
import torch  # <-- On revient à PyTorch
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# --- 1. Configuration ---
DATABASE_FILE = "dailydialog.index"
MAP_FILE = "dailydialog_map.json"

# Le "Bibliothécaire" (utilise PyTorch par défaut)
RETRIEVER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# Le "Rédacteur" (on prend Flan-T5, optimisé pour les instructions)
GENERATOR_MODEL = 'google/flan-t5-base' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du périphérique : {device}")

# --- 2. Chargement des composants RAG ---

print("Chargement du Retriever (SentenceTransformer)...")
retriever = SentenceTransformer(RETRIEVER_MODEL).to(device)

print(f"Chargement du Générateur ({GENERATOR_MODEL}) en PyTorch...")
# On charge le tokenizer de T5
tokenizer = T5Tokenizer.from_pretrained(GENERATOR_MODEL)
# On charge le modèle T5 (version PyTorch par défaut)
generator = T5ForConditionalGeneration.from_pretrained(GENERATOR_MODEL).to(device)
generator.eval() # Mettre en mode évaluation

print("Chargement de la base de données FAISS...")
index = faiss.read_index(DATABASE_FILE)
print("Chargement du mapping Q/R...")
with open(MAP_FILE, 'r', encoding='utf-8') as f:
    data_map = json.load(f)
    questions_db = data_map["questions"]
    responses_db = data_map["responses"]

print("Modèles RAG chargés.")

# --- 3. Fonctions de recherche et de génération ---

def find_context(user_query, k=3):
    """
    1. Encode la question de l'utilisateur.
    2. Cherche les k Q/R les plus proches dans la base FAISS.
    3. Formate le contexte.
    """
    query_vector = retriever.encode([user_query], convert_to_tensor=True)
    query_vector_np = query_vector.cpu().numpy().astype(np.float32)
    
    distances, indices = index.search(query_vector_np, k)
    
    context_str = ""
    for i in indices[0]:
        context_str += f"Q: {questions_db[i]}\nA: {responses_db[i]}\n"
    
    return context_str.strip()

def generate_response(user_query, context, conversation_history=""):
    """
    1. Construit le "prompt" (instruction) pour le générateur Flan-T5.
    2. Génère une réponse basée sur ce prompt et l'historique de conversation.
    """
    # Prompt unifié et clair pour Flan-T5
    if conversation_history.strip():
        prompt = f"""Answer the following question based on the conversation so far. Be natural and friendly.

Conversation:
{conversation_history}
User: {user_query}
Bot:"""
    else:
        prompt = f"""Answer this question in a friendly way:

{user_query}

Answer:"""
    
    # 1. Tokenize le prompt pour T5
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=512, 
        truncation=True
    ).to(device)
    
    # 2. Générer la réponse avec paramètres optimisés
    outputs = generator.generate(
        **inputs,
        max_new_tokens=30,        # Limite la longueur de la réponse générée
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2
    )
    
    # 3. Décoder la réponse
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# --- 4. Boucle de Chat ---
def chat():
    print("="*50)
    print("Chatbot (RAG avec Flan-T5 + PyTorch) prêt !")
    print("Tapez 'q' ou 'quit' pour arrêter.")
    print("Tapez 'clear' pour effacer l'historique de conversation.")
    print("="*50)
    
    # Liste pour stocker l'historique de la conversation
    conversation_history = []
    max_history_length = 8  # Garder les 8 derniers échanges
    
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ['q', 'quit']:
                break
            
            # Commande pour effacer l'historique
            if user_input.lower() == 'clear':
                conversation_history = []
                print("Bot: Historique de conversation effacé.")
                continue

            # 1. Formater l'historique de conversation
            history_str = ""
            for user_msg, bot_msg in conversation_history[-max_history_length:]:
                history_str += f"User: {user_msg}\nBot: {bot_msg}\n"
            
            # 2. Ne plus utiliser le contexte FAISS (cause de confusion)
            # Le modèle se base uniquement sur l'historique
            context = ""
            
            # 3. Générer la réponse
            response = generate_response(user_input, context, history_str)
            
            # 4. Nettoyer la réponse (enlever les préfixes indésirables)
            response = response.replace("Bot:", "").replace("User:", "").strip()
            
            # 5. Ajouter cet échange à l'historique
            conversation_history.append((user_input, response))
            
            print("Bot:", response)
            # print(f"  (Contexte utilisé: {context[:100]}...)\n") # Décommentez pour déboguer

        except KeyboardInterrupt:
            print("\nAu revoir !")
            break

if __name__ == "__main__":
    chat()