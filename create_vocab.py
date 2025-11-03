import json
from collections import Counter

# --- Configuration ---
VOCAB_SIZE = 10000  # Taille du vocabulaire (ajustable)
INPUTS_FILE = "inputs.txt"
TARGETS_FILE = "targets.txt"
VOCAB_FILE = "vocab.json"

# Tokens spéciaux
PAD_TOKEN = '<PAD>' # Rembourrage
START_TOKEN = '<START>' # Début de phrase
END_TOKEN = '<END>'   # Fin de phrase
UNK_TOKEN = '<UNK>' # Mot inconnu

print("Étape 1: Comptage des mots...")
word_counts = Counter()

# Lire les deux fichiers pour construire le vocabulaire
with open(INPUTS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        word_counts.update(line.split())
        
with open(TARGETS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        word_counts.update(line.split())

print(f"Trouvé {len(word_counts)} mots uniques.")

# --- Création des dictionnaires ---

# Garder les N mots les plus fréquents
most_common_words = word_counts.most_common(VOCAB_SIZE)

# 1. Dictionnaire : Mot -> Entier (word_to_int)
word_to_int = {
    PAD_TOKEN: 0,
    START_TOKEN: 1,
    END_TOKEN: 2,
    UNK_TOKEN: 3
}

# Commencer l'indexation des "vrais" mots à partir de 4
for i, (word, count) in enumerate(most_common_words):
    word_to_int[word] = i + 4

# 2. Dictionnaire : Entier -> Mot (int_to_word)
# Inverser le dictionnaire
int_to_word = {index: word for word, index in word_to_int.items()}

print(f"Taille finale du vocabulaire (avec tokens spéciaux) : {len(word_to_int)}")

# 3. Sauvegarder les deux dictionnaires dans un seul fichier JSON
vocab_data = {
    "word_to_int": word_to_int,
    "int_to_word": int_to_word
}

with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=4)

print(f"Vocabulaire sauvegardé dans {VOCAB_FILE}")