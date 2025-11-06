import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import unicodedata
import heapq # File de priorité pour le Beam Search

# --- Importation de nos modules personnalisés ---
try:
    from data_utils import Voc, MAX_LENGTH, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
    from model import EncoderRNN, LuongAttnDecoderRNN
except ImportError:
    print("ERREUR: Assure-toi que data_utils.py et model.py sont dans le dossier.")
    exit()

# --- 1. Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MAX_LENGTH != 15:
    print(f"AVERTISSEMENT: MAX_LENGTH est à {MAX_LENGTH}, mais devrait être 15.")

VOC_FILE = "voc_cornell.json"
MODEL_FILE = "chatbot_model_cornell_best.pth"

# Paramètres du modèle
attn_model = 'dot'
hidden_size = 256
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.2

# --- 2. Fonctions de normalisation (identiques) ---

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"http\S+|www\.\S+", "<url>", s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-z0-9'.!?<>-]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# --- 3. Fonctions de conversion (identiques) ---

def indexesFromSentence(voc, sentence):
    return [voc.word2index.get(word, UNK_TOKEN) for word in sentence.split(' ')] + [EOS_TOKEN]

def inputTensor(sentence, voc):
    indexes = indexesFromSentence(voc, sentence)
    lengths = torch.tensor([len(indexes)], device="cpu")
    tensor = torch.LongTensor(indexes).view(-1, 1)
    tensor = tensor.to(device)
    return tensor, lengths

# --- 4. NOUVELLE CLASSE CORRIGÉE : BeamSearchDecoder ---

# Un "nœud" pour garder la trace des hypothèses
class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.hidden = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    # Surcharge de l'opérateur "less than" pour la file de priorité
    def __lt__(self, other):
        # Score = log-probabilité normalisée par la longueur
        # On veut la *plus haute* probabilité (le score le plus élevé).
        # heapq est une "min-heap", donc le "plus petit" item est celui
        # avec le plus haut score.
        return (self.logp / float(self.leng)) > (other.logp / float(other.leng))

class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, device, beam_width=5):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.beam_width = beam_width # Largeur du faisceau (K)

    def forward(self, input_tensor, input_length, max_length=MAX_LENGTH):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            # 1. Passer l'input dans l'Encodeur
            encoder_outputs, encoder_hidden = self.encoder(input_tensor, input_length, None)

            # 2. Préparer le Décodeur
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]
            
            # 3. Créer le nœud racine (avec <SOS>)
            root_node = BeamSearchNode(
                hiddenstate=decoder_hidden,
                previousNode=None,
                wordId=SOS_TOKEN,
                logProb=0.0,
                length=1
            )
            
            # 4. Initialiser la file de priorité (le "faisceau")
            nodes = [root_node]
            heapq.heapify(nodes) # 'nodes' est le faisceau de l'étape t-1
            
            # 5. Initialiser la liste des hypothèses terminées
            completed_hypotheses = []

            # 6. Boucle de génération (jusqu'à max_length)
            for t in range(max_length):
                if not nodes:
                    break
                
                # 7. Créer le faisceau pour le *prochain* pas de temps (étape t)
                next_nodes = []

                # 8. Explorer *tous* les K nœuds du faisceau actuel
                # On vide le 'nodes' (faisceau t-1)
                while nodes:
                    current_node = heapq.heappop(nodes)

                    # 9. Si <EOS>, l'hypothèse est terminée
                    if current_node.wordid == EOS_TOKEN:
                        completed_hypotheses.append(current_node)
                        # On arrête si on a K hypothèses terminées
                        if len(completed_hypotheses) >= self.beam_width:
                            break
                        continue
                    
                    # 10. Préparer l'input pour le décodeur
                    decoder_input = torch.LongTensor([[current_node.wordid]])
                    decoder_input = decoder_input.to(self.device)

                    # 11. Appeler le décodeur (1 pas)
                    decoder_output, new_hidden = self.decoder(
                        decoder_input, current_node.hidden, encoder_outputs
                    )
                    
                    # 12. Obtenir les K meilleures prédictions (topk)
                    log_probs = F.log_softmax(decoder_output, dim=1)
                    topv, topi = log_probs.topk(self.beam_width) # [1, K]

                    # 13. Créer K nouveaux nœuds et les ajouter au *prochain* faisceau
                    for k in range(self.beam_width):
                        word_index = topi[0][k].item()
                        log_prob = topv[0][k].item()
                        
                        new_node = BeamSearchNode(
                            hiddenstate=new_hidden,
                            previousNode=current_node,
                            wordId=word_index,
                            logProb=current_node.logp + log_prob,
                            length=current_node.leng + 1
                        )
                        # On pousse dans 'next_nodes' (le futur faisceau t)
                        heapq.heappush(next_nodes, new_node) 
                
                if len(completed_hypotheses) >= self.beam_width:
                    break
                
                # 14. Élagage (Pruning) :
                # 'next_nodes' contient maintenant jusqu'à K*K candidats
                # On garde les K "plus petits" (meilleurs) pour le prochain tour
                nodes = heapq.nsmallest(self.beam_width, next_nodes)
                heapq.heapify(nodes) # 'nodes' est maintenant le faisceau t
            
            # 15. Fin de la boucle. Ajouter les nœuds restants
            completed_hypotheses.extend(nodes)
            
            # 16. Trier les hypothèses terminées par leur score
            completed_hypotheses.sort(key=lambda node: node.logp / float(node.leng), reverse=True)
            
            # 17. Choisir la *meilleure* hypothèse
            if not completed_hypotheses: # Si aucune hypothèse
                 return [EOS_TOKEN] 
                 
            best_hypothesis = completed_hypotheses[0]
            
            # 18. Reconstruire la phrase
            decoded_words_indices = []
            node = best_hypothesis
            while node.prevNode is not None:
                decoded_words_indices.append(node.wordid)
                node = node.prevNode
                
            return decoded_words_indices[::-1] # Inverser la liste

# --- 5. Boucle de Chat (identique) ---

def evaluate(searcher, voc, sentence):
    try:
        norm_sentence = normalizeString(sentence)
        input_tensor, lengths = inputTensor(norm_sentence, voc)
        
        output_indices = searcher(input_tensor, lengths)
        
        decoded_words = []
        for idx in output_indices:
            if idx == EOS_TOKEN:
                break
            word = voc.index2word.get(str(idx), UNK_TOKEN)
            decoded_words.append(word)

        decoded_words = [w for w in decoded_words if w != UNK_TOKEN]
        response = ' '.join(decoded_words)
        response = response.replace(" .", ".").replace(" ?", "?").replace(" !", "!")
        
        if not response:
            return "i'm not sure what you mean ." # Réponse de secours
            
        return response

    except Exception as e:
        print(f"Erreur lors de l'évaluation : {e}")
        return "Désolé, j'ai eu une erreur."

def chat(voc, searcher):
    print("="*50)
    print("Chatbot prêt ! (Tapez 'q' ou 'quit' pour arrêter)")
    print("="*50)
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() == 'q' or user_input.lower() == 'quit':
                break
            response = evaluate(searcher, voc, user_input)
            print("Bot:", response)
        except KeyboardInterrupt:
            print("\nAu revoir !")
            break

# --- 6. Exécution (Point d'entrée, identique) ---
if __name__ == "__main__":
    
    if not os.path.exists(VOC_FILE):
        print(f"ERREUR: Fichier vocabulaire non trouvé : {VOC_FILE}")
    else:
        print("Chargement du vocabulaire...")
        voc = Voc("cornell_corpus", VOC_FILE)
        
        print("Construction des modèles...")
        embedding = nn.Embedding(voc.num_words, hidden_size).to(device)
        encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout).to(device) # <- LIGNE CORRIGÉE
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout).to(device)

        if not os.path.exists(MODEL_FILE):
            print(f"ERREUR: Fichier modèle non trouvé : {MODEL_FILE}")
        else:
            print(f"Chargement du meilleur modèle depuis {MODEL_FILE}...")
            checkpoint = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
            encoder.load_state_dict(checkpoint['en'])
            decoder.load_state_dict(checkpoint['de'])
            embedding.load_state_dict(checkpoint['embedding'])
            
            print("Initialisation du décodeur (Beam Search, K=5)...")
            searcher = BeamSearchDecoder(encoder, decoder, device, beam_width=5)
            
            chat(voc, searcher)