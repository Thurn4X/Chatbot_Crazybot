import kagglehub
import os
import shutil

# 1. Définir le nom du sous-dossier cible
target_dir = "cornell movie-dialogs corpus"

# 2. Créer le sous-dossier s'il n'existe pas
# exist_ok=True évite une erreur si le dossier existe déjà
try:
    os.makedirs(target_dir, exist_ok=True)
    print(f"Dossier cible '{target_dir}' prêt.")
except OSError as e:
    print(f"Erreur lors de la création du dossier {target_dir}: {e}")
    # Quitter si on ne peut pas créer le dossier
    exit()

try:
    # 3. Télécharger le dataset (Kaggle le met dans son cache)
    print("Téléchargement du dataset (vers le cache Kaggle)...")
    # kagglehub.dataset_download renvoie le chemin vers le dossier dans le cache
    cache_path = kagglehub.dataset_download("rajathmc/cornell-moviedialog-corpus")
    print(f"Chemin du cache : {cache_path}")

    # 4. Copier le contenu du dossier cache vers le dossier cible
    # Nous utilisons shutil.copytree pour copier l'intégralité du répertoire
    # dirs_exist_ok=True (requis Python 3.8+) permet de fusionner le contenu
    # si target_dir existe déjà (ce que nous avons assuré à l'étape 2).
    print(f"Copie des fichiers de '{cache_path}' vers '{target_dir}'...")
    
    # Vérifier si le chemin cache existe
    if not os.path.isdir(cache_path):
        print(f"Erreur : Le chemin du cache '{cache_path}' n'est pas un dossier valide.")
    else:
        # Copier le contenu
        shutil.copytree(cache_path, target_dir, dirs_exist_ok=True)
        print(f"Copie terminée !")
        
        # Afficher le chemin absolu final
        print(f"Fichiers disponibles dans : {os.path.abspath(target_dir)}")

except Exception as e:
    print(f"Une erreur est survenue lors du téléchargement ou de la copie : {e}")