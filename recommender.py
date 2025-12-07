import clean_data
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

def charger_donnees():
    fichier_csv = 'anime_romance.csv'
    df = pd.read_csv(fichier_csv)
    df['fusion'] = df['Genres'] + " " + df['sypnopsis'] + " " + df['Name']
    return df

def preparer_modele(df):
    # On initialise le vectoriseur TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    # On transforme le texte en matrice de nombres
    tfidf_matrix = tfidf.fit_transform(df['fusion'])
    # On crée l'index inversé ICI (une seule fois) pour gagner du temps
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()
    
    return tfidf_matrix, indices

def trouver_similaires(nom_anime, matrice, indices, df):
    # Vérifie si l'anime existe dans les indices
    if nom_anime not in indices:
        return []
    # Récupère l'index de l'anime donné
    idx = indices[nom_anime]
    # Calcule les similarités cosinus entre cet anime et tous les autres
    cosine_similarities = linear_kernel(matrice[idx:idx+1], matrice).flatten()
    # Récupère les indices des 5 animes les plus similaires
    similar_indices = cosine_similarities.argsort()[:-1][-5:][::-1]
    # Récupère les détails des animes similaires
    similar_animes = df.iloc[similar_indices]
    
    return similar_animes
    

def recommander_par_wishlist(wishlist, matrice, df):
    # Récupère les indices des animes dans la wishlist
    indice_wishlist = df[df['Name'].isin(wishlist)].index
    # Si la wishlist est vide ou aucun anime n'est trouvé, retourne une liste vide
    if len(indice_wishlist) == 0:
        return []
    # Calcule le vecteur moyen de la wishlist
    vecteurs_wishlist = matrice[indice_wishlist]
    moyenne_vecteur = vecteurs_wishlist.mean(axis=0)
        
    # Calcule les similarités cosinus entre le vecteur moyen et tous les autres animes
    cosine_similarities = linear_kernel(moyenne_vecteur, matrice).flatten()
        
    # Récupère les indices des 15 animes les plus similaires
    similar_indices = cosine_similarities.argsort()[::-1][:15]
        
    # Récupère les détails des animes similaires, en excluant ceux déjà dans la wishlist
    resultats = df.iloc[similar_indices]
    recommandations = resultats[~resultats['name'].isin(wishlist)]
        
    return recommandations
    
   