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

def generate_response(user_query, context):
    """
    1. Construit le "prompt" (instruction) pour le générateur Flan-T5.
    2. Génère une réponse basée sur ce prompt.
    """
    # Flan-T5 est très bon pour suivre ce genre d'instructions
    prompt = f"""
Based on the following CONTEXT, answer the QUESTION.
Be conversational and use the style of the context.
Only output the answer, not the explanation.

CONTEXT:
{context}

QUESTION:
{user_query}

ANSWER:
"""
    
    # 1. Tokenize le prompt pour T5
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", # "pt" pour PyTorch
        max_length=512, 
        truncation=True
    ).to(device)
    
    # 2. Générer la réponse
    outputs = generator.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    # 3. Décoder la réponse
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# --- 4. Boucle de Chat ---
def chat():
    print("="*50)
    print("Chatbot (RAG avec Flan-T5 + PyTorch) prêt !")
    print("Tapez 'q' ou 'quit' pour arrêter.")
    print("="*50)
    
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ['q', 'quit']:
                break

            # 1. Trouver le contexte pertinent (Retriever)
            context = find_context(user_input, k=3)
            
            # 2. Générer la réponse basée sur le contexte (Generator)
            response = generate_response(user_input, context)
            
            print("Bot:", response)
            # print(f"  (Contexte utilisé: {context[:100]}...)\n") # Décommentez pour déboguer

        except KeyboardInterrupt:
            print("\nAu revoir !")
            break

if __name__ == "__main__":
    chat()