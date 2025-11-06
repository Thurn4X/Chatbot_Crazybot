import re
import unicodedata
import random
import os
import json
import zipfile
import sys

# --- Installation et Importation de Requests ---
try:
    import requests
except ImportError:
    print("Installation de 'requests'...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    try:
        import requests
    except ImportError:
        print("ERREUR: Impossible d'installer ou d'importer requests.")
        print("Veuillez l'installer manuellement avec : pip install requests")
        exit()

# --- 1. Configuration ---

# Source des données
DATA_URL = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
ZIP_FILE = "cornell_movie_dialogs_corpus.zip"
DATA_FOLDER = "cornell movie-dialogs corpus"
LINES_FILE = os.path.join(DATA_FOLDER, "movie_lines.txt")
CONVOS_FILE = os.path.join(DATA_FOLDER, "movie_conversations.txt")

# Fichiers de sortie
OUTPUT_VOCAB_FILE = "voc_cornell.json"
OUTPUT_PAIRS_FILE = "filtered_pairs_cornell.txt"

# Limite le nombre de paires brutes (le corpus Cornell a ~220k paires)
MAX_PAIRS_LIMIT = 500000 

# Paramètres de filtrage
MAX_LENGTH = 15
MIN_COUNT = 10 

# Tokens spéciaux
PAD_TOKEN_STR = '<PAD>'
SOS_TOKEN_STR = '<SOS>'
EOS_TOKEN_STR = '<EOS>'
UNK_TOKEN_STR = '<UNK>'

# --- 2. Classe du Vocabulaire (inchangée) ---
class Voc:
    # ... (exactement le même code que ton script précédent) ...
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {PAD_TOKEN_STR: 0, SOS_TOKEN_STR: 1, EOS_TOKEN_STR: 2, UNK_TOKEN_STR: 3}
        self.word2count = {}
        self.index2word = {0: PAD_TOKEN_STR, 1: SOS_TOKEN_STR, 2: EOS_TOKEN_STR, 3: UNK_TOKEN_STR}
        self.num_words = 4

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print(f'Mots gardés : {len(keep_words)} / {len(self.word2index) - 4} (={len(keep_words) / (len(self.word2index) - 4):.4%})')
        
        self.word2index = {PAD_TOKEN_STR: 0, SOS_TOKEN_STR: 1, EOS_TOKEN_STR: 2, UNK_TOKEN_STR: 3}
        self.word2count = {}
        self.index2word = {0: PAD_TOKEN_STR, 1: SOS_TOKEN_STR, 2: EOS_TOKEN_STR, 3: UNK_TOKEN_STR}
        self.num_words = 4
        for word in keep_words:
            self.addWord(word)

# --- 3. Fonctions de normalisation (avec apostrophe) ---
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"http\S+|www\.\S+", "<url>", s)
    s = re.sub(r"([.!?])", r" \1", s)
    # Ajout de l'apostrophe '
    s = re.sub(r"[^a-z0-9'.!?<>-]+", r" ", s) 
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# --- 4. NOUVELLES Fonctions de chargement manuel ---

def downloadAndExtractData(url, zip_name, target_folder):
    """Télécharge et extrait le zip s'il n'existe pas."""
    if os.path.exists(target_folder):
        print(f"Dossier '{target_folder}' déjà présent. Téléchargement ignoré.")
        return True

    if not os.path.exists(zip_name):
        print(f"Téléchargement de {url}...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(zip_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Téléchargement terminé.")
        except Exception as e:
            print(f"ERREUR: Impossible de télécharger {url}. {e}")
            return False
    
    print(f"Extraction de {zip_name}...")
    try:
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall()
        print(f"Extrait dans le dossier '{target_folder}'.")
        return True
    except Exception as e:
        print(f"ERREUR: Impossible d'extraire {zip_name}. {e}")
        if os.path.exists(zip_name):
             os.remove(zip_name) # Nettoyer un zip corrompu
        return False

def loadLines(filename):
    """Charge movie_lines.txt dans un dict: {lineID: text}."""
    lines = {}
    print(f"Chargement de {filename}...")
    # Le corpus utilise cet encodage spécifique
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 5:
                # Le dernier champ (parts[4]) est le texte
                lines[parts[0]] = parts[4]
    return lines

def loadConversations(filename, lines):
    """Charge movie_conversations.txt et crée les paires brutes."""
    pairs = []
    print(f"Chargement de {filename}...")
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 4:
                # Le dernier champ (parts[3]) est la liste des IDs
                # Elle est au format "['L1', 'L2', 'L3']"
                try:
                    line_ids = eval(parts[3])
                except Exception:
                    continue # Ignorer les lignes mal formées
                
                # Créer les paires (input, target)
                for i in range(len(line_ids) - 1):
                    input_line = lines.get(line_ids[i])
                    target_line = lines.get(line_ids[i+1])
                    
                    # S'assurer que les deux lignes existent
                    if input_line and target_line:
                        pairs.append([input_line, target_line])
    return pairs

def loadAndFilterPairs(max_length, max_pairs_limit):
    """Fonction principale de chargement manuel."""
    
    # 1. Télécharger et extraire
    if not downloadAndExtractData(DATA_URL, ZIP_FILE, DATA_FOLDER):
        return [], Voc("default")

    # 2. Charger les lignes
    lines = loadLines(LINES_FILE)

    # 3. Charger les conversations (qui crée les paires brutes)
    all_pairs_raw = loadConversations(CONVOS_FILE, lines)
    
    print(f"\nNombre total de paires brutes extraites : {len(all_pairs_raw)}")
    
    # 4. Limiter si nécessaire (pour accélérer)
    if len(all_pairs_raw) > max_pairs_limit:
        print(f"Limitation à {max_pairs_limit} paires brutes pour le traitement.")
        all_pairs_raw = all_pairs_raw[:max_pairs_limit]

    # 5. Normaliser les paires
    print("Normalisation des paires...")
    all_pairs_normalized = []
    for pair in all_pairs_raw:
        norm_input = normalizeString(pair[0])
        norm_target = normalizeString(pair[1])
        if norm_input and norm_target:
            all_pairs_normalized.append([norm_input, norm_target])
            
    print(f"Paires après normalisation (non vides) : {len(all_pairs_normalized)}")

    # --- Affichage des échantillons ---
    print("\n--- Échantillons de données (brut vs normalisé) ---")
    for i in range(5):
        idx = random.randint(0, len(all_pairs_normalized) - 1)
        # On affiche la version normalisée cette fois
        print(f"  Paire {idx}: {all_pairs_normalized[idx][0]} -> {all_pairs_normalized[idx][1]}")
    print("---")
    # --- Fin affichage ---

    # 6. Filtrer par longueur et remplir le Vocabulaire
    print(f"Filtrage des paires par longueur (MAX_LENGTH={max_length})...")
    filtered_pairs = []
    voc = Voc(DATA_FOLDER)
    for pair in all_pairs_normalized:
        if (len(pair[0].split(' ')) < max_length and 
            len(pair[1].split(' ')) < max_length):
            
            filtered_pairs.append(pair)
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
            
    return filtered_pairs, voc

# --- 5. Fonction de filtrage du vocabulaire (inchangée) ---
def trimRareWords(voc, pairs, min_count):
    # ... (exactement le même code que ton script précédent) ...
    print(f"\nFiltrage du vocabulaire (MIN_COUNT={min_count})...")
    voc.trim(min_count)
    keep_pairs = []
    
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False; break
        
        if keep_input:
            for word in output_sentence.split(' '):
                if word not in voc.word2index:
                    keep_output = False; break
                    
        if keep_input and keep_output:
            keep_pairs.append(pair)
            
    print(f"Taille après trimming : {len(keep_pairs)} paires (sur {len(pairs)})")
    return keep_pairs

# --- 6. Script principal d'exécution ---
if __name__ == "__main__":
    
    # 1. Charger, filtrer, et créer le vocabulaire
    pairs, voc = loadAndFilterPairs(MAX_LENGTH, MAX_PAIRS_LIMIT)
    print(f"Nombre de paires après filtrage par longueur : {len(pairs)}")
    
    # 2. Filtrer les mots rares
    final_pairs = trimRareWords(voc, pairs, MIN_COUNT)
    
    print("\n" + "="*40)
    print(f"Taille finale du vocabulaire : {voc.num_words} mots")
    print(f"Nombre final de paires (propres) : {len(final_pairs)}")

    # --- NOUVEAU BLOC DE VÉRIFICATION ---
    print("\n" + "="*40)
    print("🔍 VÉRIFICATION : 20 ÉCHANTILLONS DES PAIRES FINALES")
    print("   (C'est ce que le chatbot va apprendre)")
    print("="*40 + "\n")
    
    # S'assurer qu'il y a assez de paires à échantillonner
    num_samples = 20
    if len(final_pairs) > num_samples:
        samples = random.sample(final_pairs, num_samples)
    else:
        samples = final_pairs # Afficher tout si on a moins de 20 paires

    for i, pair in enumerate(samples):
        print(f"  {i+1:2d}: {pair[0]}")
        print(f"   -> {pair[1]}\n")

    # --- 3. SAUVEGARDE DES FICHIERS ---
    print(f"\nSauvegarde du vocabulaire dans {OUTPUT_VOCAB_FILE}...")
    voc_data = {
        "name": voc.name, 
        "word2index": voc.word2index,
        "index2word": voc.index2word, 
        "num_words": voc.num_words
    }
    with open(OUTPUT_VOCAB_FILE, 'w', encoding='utf-8') as f:
        json.dump(voc_data, f, indent=4)

    print(f"Sauvegarde des paires nettoyées dans {OUTPUT_PAIRS_FILE}...")
    with open(OUTPUT_PAIRS_FILE, 'w', encoding='utf-8') as f:
        for pair in final_pairs:
            f.write(f"{pair[0]}\t{pair[1]}\n")

    print("\n--- Terminé! (Données Cornell sauvegardées) ---")
    print(f"Fichiers générés : {OUTPUT_VOCAB_FILE} et {OUTPUT_PAIRS_FILE}")