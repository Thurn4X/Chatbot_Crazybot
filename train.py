import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Attention, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint # <-- Nouvel import

# --- 1. Configuration et Chargement ---
print("Étape 1: Chargement des données et du vocabulaire...")
VOCAB_FILE = "vocab.json"
INPUTS_NPY = "inputs.npy"
TARGETS_NPY = "targets.npy"

with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
word_to_int = vocab_data["word_to_int"]

inputs_padded = np.load(INPUTS_NPY)
targets_padded = np.load(TARGETS_NPY)

# --- 2. Paramètres du Modèle ---
VOCAB_SIZE = len(word_to_int)
MAX_SEQ_LEN = inputs_padded.shape[1] 
EMBEDDING_DIM = 128
RNN_UNITS = 256

print(f"Taille Vocabulaire: {VOCAB_SIZE}, Longueur Séquence: {MAX_SEQ_LEN}")

# --- 3. Préparation "Teacher Forcing" ---
print("Étape 2: Préparation des données (Teacher Forcing)...")
encoder_input_data = inputs_padded
decoder_input_data = targets_padded[:, :-1]
decoder_target_data = targets_padded[:, 1:]

print(f"Shape Entrée Encodeur: {encoder_input_data.shape}")
print(f"Shape Entrée Décodeur: {decoder_input_data.shape}")
print(f"Shape Sortie Décodeur: {decoder_target_data.shape}")

# --- 4. Définition de l'Architecture (Modèle d'entraînement) ---
print("Étape 3: Construction de l'architecture...")

# ... (Identique à build_model.py) ...
encoder_inputs = Input(shape=(None,), name='encoder_input')
enc_embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)
enc_emb_output = enc_embedding_layer(encoder_inputs)
encoder_lstm = LSTM(RNN_UNITS, return_sequences=True, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb_output)
encoder_states = [state_h, state_c] 

decoder_inputs = Input(shape=(None,), name='decoder_input')
dec_embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)
dec_emb_output = dec_embedding_layer(decoder_inputs)
decoder_lstm = LSTM(RNN_UNITS, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_lstm_outputs, _, _ = decoder_lstm(dec_emb_output, initial_state=encoder_states)

attention_layer = Attention(name='attention_layer')
context_vector, _ = attention_layer(
    [decoder_lstm_outputs, encoder_outputs], 
    return_attention_scores=True
)
decoder_combined_output = Concatenate(axis=-1, name='concat_attention_lstm')(
    [decoder_lstm_outputs, context_vector]
)
output_layer = TimeDistributed(
    Dense(VOCAB_SIZE, activation='softmax'), 
    name='output_layer'
)
decoder_outputs = output_layer(decoder_combined_output)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='seq2seq_attention')

# --- 5. Compilation ---
print("Étape 4: Compilation du modèle...")
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# --- 6. NOUVEAU: Paramètres d'Entraînement ---

BATCH_SIZE = 512 # on réduira à 64 plus tard
EPOCHS = 10  # on augmentera à 50 plus tard


# --- 7. NOUVEAU: Callback de Sauvegarde ---

# C'est la partie la plus importante.
# L'entraînement sera long. Si ça plante, on ne veut pas tout perdre.
# On sauvegarde le "meilleur" modèle (basé sur la val_loss)
# au format Keras natif.
checkpoint_path = "chatbot_model.keras" 

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False, # Sauvegarder le modèle complet (archi + poids)
    save_best_only=True,     # Ne garder que le meilleur
    monitor='val_loss',      # La métrique à surveiller
    mode='min',              # On veut minimiser la 'loss'
    verbose=1                # Afficher un message quand on sauvegarde
)

# --- 8. NOUVEAU: Entraînement ---
print("\n--- ÉTAPE 5: DÉBUT DE L'ENTRAÎNEMENT ---")
print(f"Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")
print("Surveillez la 'val_loss'. Elle doit diminuer.")

history = model.fit(
    [encoder_input_data, decoder_input_data],  # Nos deux entrées
    decoder_target_data,                       # Notre sortie
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2, # Garder 20% des données pour la validation
    callbacks=[checkpoint_callback]
)

print("\n--- Entraînement terminé! ---")
print(f"Le meilleur modèle a été sauvegardé dans {checkpoint_path}")