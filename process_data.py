import re
import ast  # Pour évaluer la liste de string en tant que liste Python

# --- 1. Configuration ---
CORPUS_DIR = "./cornell movie-dialogs corpus"

LINES_FILE = f"{CORPUS_DIR}/movie_lines.txt"
CONV_FILE = f"{CORPUS_DIR}/movie_conversations.txt"

# Fichiers de sortie
OUTPUT_INPUTS = "inputs.txt"
OUTPUT_TARGETS = "targets.txt"

MIN_LEN = 3   # Nombre de mots minimum
MAX_LEN = 20  # Nombre de mots maximum

# --- 2. Fonction de nettoyage ---

def normalize_text(text):
    """
    Nettoie et normalise une chaîne de texte.
    - Met en minuscule
    - Ajoute des espaces autour de la ponctuation
    - Supprime les caractères non-désirés
    - Réduit les espaces multiples
    """
    text = text.lower()
    
    # Ajoute un espace avant/après la ponctuation pour la "tokéniser"
    # "ça va?" -> "ça va ?"
    text = re.sub(r"([?!.,])", r" \1", text)
    
    # Garde seulement les lettres, chiffres et ponctuation de base
    # (Supprime ♪, <...>, etc.)
    text = re.sub(r"[^a-z0-9'?!., ]", "", text)
    
    # Remplace les espaces multiples par un seul
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# --- 3. Étape 1 : Charger les lignes (ID -> Texte) ---

print("Étape 1: Chargement de movie_lines.txt...")
lines_dict = {}
# Note: Ce corpus utilise un encodage spécifique
with open(LINES_FILE, 'r', encoding='iso-8859-1') as f:
    for line in f:
        # Format: L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
        parts = line.split(" +++$+++ ")
        if len(parts) == 5:
            line_id = parts[0]
            text = parts[4].strip()  # Le texte est le dernier élément
            lines_dict[line_id] = text

print(f"Chargé {len(lines_dict)} lignes.")

# --- 4. Étape 2 & 3 & 4 : Traiter les conversations, nettoyer et écrire ---

print("Étape 2: Traitement des conversations et écriture des paires...")
pairs_created = 0

# On ouvre les deux fichiers de sortie en même temps
with open(OUTPUT_INPUTS, 'w', encoding='utf-8') as f_in, \
     open(OUTPUT_TARGETS, 'w', encoding='utf-8') as f_out:
    
    with open(CONV_FILE, 'r', encoding='iso-8859-1') as f:
        for line in f:
            # Format: u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196']
            parts = line.split(" +++$+++ ")
            if len(parts) == 4:
                # Le dernier élément est la liste des ID de lignes
                line_ids_str = parts[3].strip()
                
                # ast.literal_eval convertit la string "['L1', 'L2']"
                # en une vraie liste Python ['L1', 'L2']
                try:
                    line_ids = ast.literal_eval(line_ids_str)
                except (ValueError, SyntaxError):
                    continue # Ignore les lignes mal formées
                
                # Créer les paires (input -> target)
                for i in range(len(line_ids) - 1):
                    input_id = line_ids[i]
                    target_id = line_ids[i+1]
                    
                    # Vérifier que les ID existent dans notre dictionnaire
                    if input_id in lines_dict and target_id in lines_dict:
                        input_text = normalize_text(lines_dict[input_id])
                        target_text = normalize_text(lines_dict[target_id])
                        
                        # --- FILTRAGE ---
                        input_words = input_text.split()
                        target_words = target_text.split()
                        
                        if (MIN_LEN <= len(input_words) <= MAX_LEN) and \
                           (MIN_LEN <= len(target_words) <= MAX_LEN) and \
                           input_text and target_text:
                            
                            # Écrire dans les fichiers
                            f_in.write(input_text + "\n")
                            f_out.write(target_text + "\n")
                            pairs_created += 1

print("---")
print("Terminé !")
print(f"{pairs_created} paires (Input/Target) ont été créées.")
print(f"Fichiers générés : {OUTPUT_INPUTS} et {OUTPUT_TARGETS}")