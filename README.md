Projet de chatbot 

La version finale du chatbot de ce projet est le Chatbot RAG. Il est conseillé de run ce dernier, mais deux autres chatbots expérimentaux sont disponibles: BERT-GPT2 et GRU-GRU+AttentionLuong ()

Pour le Bot Rag:
- cd RAG_chatbot
- pip install sentence-transformers faiss-cpu numpy
- python build_database.py (Attention cette etape peut prendre quelques minutes)
- python chat_rag_pytorch.py
- discuter

Pour le bot BERT-GPT:
- le dossier du modèle est disponible dans les releases en tant que pre release: Modèle BERT-GPT2 finetuné v2
- télécharger le dossier
- python chat_bert_gpt_2.py
- discuter
  

Pour Le bot GRU/GRU+ attention
- pip install numpy, tensorflow
- lancer chat_cornell.py
- discuter
- optionnel: regarder le notebook utilisé pour l'entrainer: https://www.kaggle.com/code/thurn4x/crazybot-cornell

![alt text](image-1.png)

