import torch
import json
import itertools
import os # Ajout de 'os' pour la vérification des fichiers

# --- Constantes de Tokens ---
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3 

# --- MODIFICATION IMPORTANTE ---
# Doit correspondre à la valeur de 1_prepare_ubuntu_final.py
MAX_LENGTH = 15 
# --- FIN MODIFICATION ---


# --- 1. Logique de chargement des données ---

class Voc:
    """Classe simple pour RECHARGER le voc.json"""
    def __init__(self, name, voc_file="/kaggle/input/cornell-inputs/voc_cornell.json"): # Mis à jour pour le nom par défaut
        self.name = name
        self.voc_file = voc_file
        self.load_data()

    def load_data(self):
        try:
            with open(self.voc_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.word2index = data["word2index"]
                self.index2word = data["index2word"]
                # --- CETTE LIGNE EST CRUCIALE ---
                self.num_words = data["num_words"]
                # --- FIN ---
        except FileNotFoundError:
            print(f"ERREUR: Fichier vocabulaire non trouvé : {self.voc_file}")
            print("Veuillez d'abord exécuter le script de préparation des données.")
            exit()
        except KeyError:
            print(f"ERREUR: Clé 'num_words' non trouvée dans {self.voc_file}.")
            print("Veuillez ré-exécuter le script '1_prepare_data.py'.")
            exit()


def load_pairs(filename):
    print(f"Lecture des paires depuis {filename}...")
    pairs = []
    if not os.path.exists(filename):
        print(f"ERREUR: Fichier de paires non trouvé : {filename}")
        print("Veuillez d'abord exécuter le script de préparation des données.")
        exit()
        
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                pairs.append(parts)
    return pairs

# --- 2. Fonctions de préparation des Batchs ---

def indexesFromSentence(voc, sentence):
    # Utilise UNK_TOKEN pour les mots qui ont été filtrés (ex: pseudos)
    return [voc.word2index.get(word, UNK_TOKEN) for word in sentence.split(' ')] + [EOS_TOKEN]

def zeroPadding(l, fillvalue=PAD_TOKEN):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_TOKEN):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_TOKEN:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Fonctions "Wrapper" (enveloppe) pour convertir les paires en tenseurs

def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Fonction principale pour l'extérieur : prend des paires, retourne des tenseurs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len