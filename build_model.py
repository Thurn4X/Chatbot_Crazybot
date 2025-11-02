import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Model
# ... autres imports
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Attention, Concatenate

# --- 1. Configuration et Chargement ---

print("Étape 1: Chargement des données et du vocabulaire...")

VOCAB_FILE = "vocab.json"
INPUTS_NPY = "inputs.npy"
TARGETS_NPY = "targets.npy"

# Charger le vocabulaire pour connaître la taille
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
word_to_int = vocab_data["word_to_int"]

# Charger les données numérisées
inputs_padded = np.load(INPUTS_NPY)
targets_padded = np.load(TARGETS_NPY)

# --- 2. Paramètres du Modèle ---

VOCAB_SIZE = len(word_to_int)
# MAX_SEQ_LEN doit être la 2ème dimension de vos fichiers .npy
# (ex: 22 si MAX_LEN_BASE=20 + 2 tokens)
MAX_SEQ_LEN = inputs_padded.shape[1] 

EMBEDDING_DIM = 128  # Taille des vecteurs de mots (hyperparamètre) on le mutlipliera par deux plus tard
RNN_UNITS = 256      # Nombre de neurones dans le LSTM (hyperparamètre) on le mutlipliera par deux plus tard

print(f"Taille Vocabulaire: {VOCAB_SIZE}, Longueur Séquence: {MAX_SEQ_LEN}")

# --- 3. Préparation "Teacher Forcing" ---

print("Étape 2: Préparation des données (Teacher Forcing)...")

# 1. Entrée de l'encodeur (la question)
# ex: [<START>, 'comment', 'vas', 'tu', '?', <END>, <PAD>, ...]
encoder_input_data = inputs_padded

# 2. Entrée du décodeur (la réponse, décalée)
# ex: [<START>, 'je', 'vais', 'bien', <END>, <PAD>, <PAD>, ...]
# On prend tout sauf le dernier token (qui sera un <PAD> ou <END>)
decoder_input_data = targets_padded[:, :-1]

# 3. Sortie du décodeur (ce qu'il doit prédire)
# ex: ['je', 'vais', 'bien', <END>, <PAD>, <PAD>, <PAD>, ...]
# On prend tout sauf le premier token (<START>)
decoder_target_data = targets_padded[:, 1:]

print(f"Shape Entrée Encodeur: {encoder_input_data.shape}")
print(f"Shape Entrée Décodeur: {decoder_input_data.shape}")
print(f"Shape Sortie Décodeur: {decoder_target_data.shape}")


# --- 4. Définition de l'Architecture (Modèle d'entraînement) ---
# Nous utilisons l'API Fonctionnelle de Keras

print("Étape 3: Construction de l'architecture...")

# --- A. L'ENCODEUR ---

# L'encodeur reçoit la séquence d'entrée
encoder_inputs = Input(shape=(None,), name='encoder_input')

# Couche d'Embedding (transforme les entiers en vecteurs denses)
# mask_zero=True indique d'ignorer le padding (index 0)
enc_embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)
enc_emb_output = enc_embedding_layer(encoder_inputs)

# Couche LSTM
# return_sequences=True : retourne la sortie à *chaque* étape (pour l'attention)
# return_state=True : retourne l'état caché (h) et l'état de cellule (c)
encoder_lstm = LSTM(RNN_UNITS, return_sequences=True, return_state=True, name='encoder_lstm')
# encoder_outputs : sorties de *toutes* les étapes (pour l'attention)
# state_h, state_c : état final (pour initialiser le décodeur)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb_output)
encoder_states = [state_h, state_c] # On garde l'état final

# --- B. LE DÉCODEUR ---

# Le décodeur reçoit la séquence cible (décalée)
decoder_inputs = Input(shape=(None,), name='decoder_input')

# Couche d'Embedding (la même dimension que l'encodeur)
dec_embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)
dec_emb_output = dec_embedding_layer(decoder_inputs)

# Couche LSTM
# return_sequences=True : le décodeur doit prédire un mot à chaque étape
decoder_lstm = LSTM(RNN_UNITS, return_sequences=True, return_state=True, name='decoder_lstm')
# On initialise l'état du décodeur avec l'état final de l'encodeur
decoder_lstm_outputs, _, _ = decoder_lstm(dec_emb_output, initial_state=encoder_states)

# --- C. LE MÉCANISME D'ATTENTION (Bahdanau) ---
# L'attention compare les sorties de l'encodeur (encoder_outputs)
# avec la sortie actuelle du décodeur (decoder_lstm_outputs)
attention_layer = Attention(name='attention_layer')
# 'context_vector' est le résumé "pondéré" de l'entrée, 
# spécifique à cette étape de décodage
context_vector, attention_weights = attention_layer(
    [decoder_lstm_outputs, encoder_outputs], 
    return_attention_scores=True
)

# On concatène la sortie du LSTM et le vecteur de contexte
# C'est ce qui donne au décodeur l'info sur "où regarder"
# Ligne 111 (la corrigée)
decoder_combined_output = Concatenate(axis=-1, name='concat_attention_lstm')(
    [decoder_lstm_outputs, context_vector]
)

# --- D. La Couche de Sortie (Tête de prédiction) ---
# Une couche Dense pour prédire le mot suivant
# TimeDistributed applique la même couche Dense à chaque étape de la séquence
# L'activation 'softmax' donne une probabilité pour chaque mot du vocabulaire
output_layer = TimeDistributed(
    Dense(VOCAB_SIZE, activation='softmax'), 
    name='output_layer'
)
decoder_outputs = output_layer(decoder_combined_output)

# --- E. Création du Modèle Final ---
model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='seq2seq_attention')

# --- 5. Compilation ---
print("Étape 4: Compilation du modèle...")

# 'sparse_categorical_crossentropy' est utilisé car nos cibles (decoder_target_data)
# sont des *entiers* (ex: 5) et non des vecteurs one-hot (ex: [0,0,0,0,1,0...])
# C'est beaucoup plus efficace en mémoire.
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# Sauvegarder un schéma du modèle (optionnel, mais utile)
try:
    tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True)
    print("Architecture sauvegardée dans 'model_architecture.png'")
except ImportError:
    print("Impossible de générer l'image du modèle (Graphviz non installé ?)")