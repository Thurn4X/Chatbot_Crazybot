import os
import numpy as np
import faiss  # Pour la base de données vectorielle
from sentence_transformers import SentenceTransformer
import json
import zipfile # <-- Ajouté

# --- 1. Configuration ---
# Fichiers de données (DailyDialog local)
LOCAL_ZIP_FILE = "train.zip"
LOCAL_TXT_FILE = "train/dialogues_train.txt" 

# Modèle "Retriever"
RETRIEVER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Fichiers de sortie
DATABASE_FILE = "dailydialog.index"
MAP_FILE = "dailydialog_map.json"

# --- 2. Chargement et Formatage des Données (MODIFIÉ) ---
def load_and_format_data(zip_path, txt_path):
    """
    Charge les données depuis le fichier .zip local de DailyDialog.
    """
    
    # Étape 1 : Décompresser le .zip s'il n'est pas déjà décompressé
    if not os.path.exists(txt_path):
        print(f"Fichier {txt_path} non trouvé. Tentative d'extraction de {zip_path}...")
        if not os.path.exists(zip_path):
            print(f"ERREUR: Fichier '{zip_path}' non trouvé.")
            print("Veuillez l'uploader dans votre session Colab/Kaggle.")
            return [], [] # Retourne des listes vides
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("./") # Extrait dans le dossier courant
            print(f"Fichier {zip_path} extrait avec succès.")
        except Exception as e:
            print(f"ERREUR lors de l'extraction du zip : {e}")
            return [], []

    # Étape 2 : Lire le fichier .txt extrait
    print(f"Chargement du fichier local '{txt_path}'...")
    questions = []
    reponses = []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Chaque ligne est une conversation complète, séparée par "__eou__"
            conversation = line.strip().split('__eou__')
            
            for i in range(len(conversation) - 1):
                input_text = conversation[i].strip()
                label_text = conversation[i+1].strip()
                
                # Filtre de sécurité
                if input_text and label_text:
                    questions.append(input_text)
                    reponses.append(label_text)

    print(f"Créé {len(questions)} paires de conversation.")
    return questions, reponses

# --- 3. Script principal ---
if __name__ == "__main__":
    
    # 1. Charger les données
    questions, reponses = load_and_format_data(LOCAL_ZIP_FILE, LOCAL_TXT_FILE)
    
    if not questions:
        print("Aucune donnée chargée. Arrêt.")
        exit()
        
    # 2. Charger le modèle "Retriever"
    print(f"Chargement du modèle retriever '{RETRIEVER_MODEL}'...")
    retriever_model = SentenceTransformer(RETRIEVER_MODEL)

    # 3. Créer les "Vecteurs" (Embeddings)
    print("Encodage des questions en vecteurs... (cela peut prendre quelques minutes)")
    question_vectors = retriever_model.encode(questions, show_progress_bar=True)
    
    # 4. Créer la base de données FAISS
    print("Construction de l'index FAISS...")
    dimension = question_vectors.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(question_vectors.astype(np.float32))

    print(f"Index FAISS construit avec {index.ntotal} vecteurs.")

    # 5. Sauvegarder les fichiers
    print(f"Sauvegarde de l'index dans {DATABASE_FILE}...")
    faiss.write_index(index, DATABASE_FILE)

    print(f"Sauvegarde du mapping (index -> Q/R) dans {MAP_FILE}...")
    # On sauvegarde les questions ET les réponses
    data_to_save = {"questions": questions, "responses": reponses}
    with open(MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f)

    print("\n--- Base de données RAG construite avec succès ! ---")