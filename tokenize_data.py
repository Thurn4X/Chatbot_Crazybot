import json
import numpy as np
# Nous utilisons Keras seulement pour sa fonction de padding, très pratique
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
VOCAB_FILE = "vocab.json"
INPUTS_FILE = "inputs.txt"
TARGETS_FILE = "targets.txt"

OUTPUT_INPUTS_NPY = "inputs.npy"
OUTPUT_TARGETS_NPY = "targets.npy"

# Doit correspondre à la valeur du script précédent
# (MAX_LEN que vous aviez mis à 20)
MAX_LEN_BASE = 20 
# +2 car nous ajoutons <START> et <END> à chaque phrase
MAX_SEQ_LEN = MAX_LEN_BASE + 2 

# --- 1. Charger le vocabulaire ---
print("Étape 1: Chargement du vocabulaire...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
word_to_int = vocab_data["word_to_int"]

# Récupérer l'index du token <UNK>
UNK_INDEX = word_to_int[r'<UNK>']
START_INDEX = word_to_int[r'<START>']
END_INDEX = word_to_int[r'<END>']

# --- 2. Fonction de conversion ---
def text_to_sequence(text, word_to_int_map):
    """Convertit une phrase en une liste d'entiers + <START>/<END>"""
    seq = [START_INDEX] # Commencer par <START>
    
    for word in text.split():
        # .get(word, UNK_INDEX) :
        # 1. Essaie de trouver le mot
        # 2. S'il n'est pas trouvé, utilise UNK_INDEX
        seq.append(word_to_int_map.get(word, UNK_INDEX))
        
    seq.append(END_INDEX) # Terminer par <END>
    return seq

# --- 3. Traitement des fichiers ---
print(f"Étape 2: Conversion de {INPUTS_FILE}...")
inputs_sequences = []
with open(INPUTS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        inputs_sequences.append(text_to_sequence(line, word_to_int))

print(f"Étape 3: Conversion de {TARGETS_FILE}...")
targets_sequences = []
with open(TARGETS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        targets_sequences.append(text_to_sequence(line, word_to_int))

# --- 4. Padding ---


print(f"Étape 4: Padding des séquences (longueur max = {MAX_SEQ_LEN})...")

# 'post' signifie qu'on ajoute le padding (0) à la FIN de la phrase
# 'truncating' signifie qu'on coupe la FIN si c'est trop long
inputs_padded = pad_sequences(
    inputs_sequences, 
    maxlen=MAX_SEQ_LEN, 
    padding='post', 
    truncating='post'
)

targets_padded = pad_sequences(
    targets_sequences, 
    maxlen=MAX_SEQ_LEN, 
    padding='post', 
    truncating='post'
)

# --- 5. Sauvegarde ---
print("Étape 5: Sauvegarde des fichiers .npy...")
np.save(OUTPUT_INPUTS_NPY, inputs_padded)
np.save(OUTPUT_TARGETS_NPY, targets_padded)

print("---")
print("Terminé !")
print(f"Données d'entrée sauvegardées dans : {OUTPUT_INPUTS_NPY} (Shape: {inputs_padded.shape})")
print(f"Données cibles sauvegardées dans : {OUTPUT_TARGETS_NPY} (Shape: {targets_padded.shape})")