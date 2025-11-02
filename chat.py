import numpy as np
import json
import re
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Configuration et Chargement ---

print("Chargement du modèle et du vocabulaire...")

VOCAB_FILE = "vocab.json"
MODEL_FILE = "chatbot_model.keras"

# Charger le vocabulaire
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
word_to_int = vocab_data["word_to_int"]
int_to_word = vocab_data["int_to_word"]


VOCAB_SIZE = len(word_to_int) # Récupérer les constantes depuis le vocabulaire/modèle

MAX_SEQ_LEN = 22 # Doit correspondre à la valeur de tokenize_data.py (ex: 22)

RNN_UNITS = 256 # Utiliser le meme nombre quand dans la création du modèle et l'entrainement

# Charger le modèle d'entraînement complet
# On le charge pour pouvoir extraire ses couches
trained_model = load_model(MODEL_FILE)

# --- 2. Fonctions de Prétraitement (Identiques à avant) ---

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"([?!.,])", r" \1", text)
    text = re.sub(r"[^a-z0-9'?!., ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Tokens spéciaux
START_TOKEN_IDX = word_to_int['<START>']
END_TOKEN_IDX = word_to_int['<END>']
UNK_TOKEN_IDX = word_to_int['<UNK>']

# --- 3. Reconstruction des Modèles d'Inférence ---

print("Reconstruction des modèles d'inférence...")

# --- A. MODÈLE ENCODEUR D'INFÉRENCE ---
# Prend la phrase d'entrée -> Sort les états et les sorties de l'encodeur

# L'entrée est la même que celle du modèle entraîné
encoder_inputs = trained_model.input[0]

# Les sorties sont celles de la couche LSTM de l'encodeur
# (on a besoin des 'outputs' pour l'attention, et des 'states' pour le décodeur)
encoder_outputs, state_h, state_c = trained_model.get_layer('encoder_lstm').output

# Définir le modèle
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])
print("Modèle Encodeur OK.")

# --- B. MODÈLE DÉCODEUR D'INFÉRENCE ---
# Prend (Input du mot N), (États précédents), (Sorties de l'encodeur)
# -> Sort (Prédiction du mot N+1), (Nouveaux états)

# Inputs pour le décodeur
inf_dec_inputs = Input(shape=(None,), name='inf_dec_inputs') # Input (1 mot à la fois)
inf_enc_outputs = Input(shape=(None, RNN_UNITS), name='inf_enc_outputs') # Sorties encodeur (pour attention)
inf_state_h = Input(shape=(RNN_UNITS,), name='inf_state_h') # État h
inf_state_c = Input(shape=(RNN_UNITS,), name='inf_state_c') # État c
inf_decoder_states = [inf_state_h, inf_state_c]

# Récupérer les couches du modèle entraîné
# 'embedding_1' est le nom auto-généré par Keras pour la *deuxième* couche Embedding
dec_emb_layer = trained_model.get_layer('embedding_1') 
dec_lstm = trained_model.get_layer('decoder_lstm')
attention = trained_model.get_layer('attention_layer')
concat = trained_model.get_layer('concat_attention_lstm')
output_layer = trained_model.get_layer('output_layer')

# Recâbler les couches pour l'inférence
dec_emb = dec_emb_layer(inf_dec_inputs)
lstm_out, out_h, out_c = dec_lstm(dec_emb, initial_state=inf_decoder_states)
out_decoder_states = [out_h, out_c]

context, _ = attention(
    [lstm_out, inf_enc_outputs], 
    return_attention_scores=True
)
combined = concat([lstm_out, context])
output = output_layer(combined)

decoder_model = Model(
    [inf_dec_inputs, inf_enc_outputs, inf_state_h, inf_state_c],
    [output, out_decoder_states]
)
print("Modèle Décodeur OK.")


# --- 4. Fonction de "Chat" (La boucle d'inférence) ---

def decode_sequence(input_text):
    # 1. Nettoyer et tokeniser l'entrée
    input_seq = [START_TOKEN_IDX]
    for word in normalize_text(input_text).split():
        input_seq.append(word_to_int.get(word, UNK_TOKEN_IDX))
    input_seq.append(END_TOKEN_IDX)
    
    # 2. Padder la séquence
    input_seq = pad_sequences([input_seq], maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

    # 3. Encoder l'entrée pour obtenir les états
    enc_out, enc_h, enc_c = encoder_model.predict(input_seq, verbose=0)
    
    # 4. Initialiser la boucle de décodage
    # On commence avec le token <START>
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = START_TOKEN_IDX
    
    stop_condition = False
    decoded_sentence = ""
    
    # 5. Boucle de génération (mot par mot)
    while not stop_condition:
        # Prédire le mot suivant
        output_tokens, [enc_h, enc_c] = decoder_model.predict(
            [target_seq, enc_out, enc_h, enc_c],
            verbose=0
        )
        
        # Obtenir l'ID du mot avec la plus haute probabilité (Argmax)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        # Convertir l'ID en mot
        # Utiliser str() car nos clés JSON (int_to_word) sont des strings
        sampled_word = int_to_word.get(str(sampled_token_index), '<UNK>')

        # 6. Condition d'arrêt
        # Si on prédit <END> ou si la phrase est trop longue
        if (sampled_word == '<END>' or len(decoded_sentence.split()) > MAX_SEQ_LEN):
            stop_condition = True
        else:
            decoded_sentence += " " + sampled_word
            
        # 7. Mettre à jour le prochain input du décodeur
        target_seq[0, 0] = sampled_token_index
        
    return decoded_sentence.strip()

# --- 5. Boucle Principale ---

print("\n--- Chatbot Prêt! (tapez 'exit' pour quitter) ---")
while True:
    input_text = input("> ")
    if input_text.lower() == 'exit':
        break
    
    response = decode_sequence(input_text)
    print("Bot:", response)