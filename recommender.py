import clean_data
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

def charger_donnees():
    fichier_csv = 'anime_romance.csv'
    df = pd.read_csv(fichier_csv)
    df['fusion'] = df['Genres'] + " " + df['sypnopsis']
    return df

def preparer_modele(df):
    # On initialise le vectoriseur TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    # On transforme le texte en matrice de nombres
    tfidf_matrix = tfidf.fit_transform(df['fusion'])
    # On crée l'index inversé ICI (une seule fois) pour gagner du temps
    indices = pd.Series(df.index, index=df['Name']).drop_duplicates()
    
    return tfidf_matrix, indices

    
def recommander_par_wishlist(wishlist, matrice, df):
    # Récupère les indices des animes dans la wishlist
    indice_wishlist = df[df['Name'].isin(wishlist)].index
    # Si la wishlist est vide ou aucun anime n'est trouvé, retourne une liste vide
    if len(indice_wishlist) == 0:
        return []
    # Calcule le vecteur moyen de la wishlist
    vecteurs_wishlist = matrice[indice_wishlist]
    moyenne_vecteur = np.asarray(vecteurs_wishlist.mean(axis=0))
        
    # Calcule les similarités cosinus entre le vecteur moyen et tous les autres animes
    cosine_similarities = linear_kernel(moyenne_vecteur, matrice).flatten()
        
    # Récupère les indices des animes les plus similaires
    similar_indices = cosine_similarities.argsort()[::-1]
        
    # Récupère les détails des animes similaires, en excluant ceux déjà dans la wishlist
    resultats = df.iloc[similar_indices]
    recommandations = resultats[~resultats['Name'].isin(wishlist)]
        
    return recommandations

def recommander_par_deja_vus(watched, matrice, df):
    # Récupère les indices des animes déjà vus
    indice_watched = df[df['Name'].isin(watched)].index
    # Si la liste est vide ou aucun anime n'est trouvé, retourne une liste vide
    if len(indice_watched) == 0:
        return pd.DataFrame()
    # Calcule le vecteur moyen des animes déjà vus
    vecteurs_watched = matrice[indice_watched]
    moyenne_vecteur = np.asarray(vecteurs_watched.mean(axis=0))
        
    # Calcule les similarités cosinus entre le vecteur moyen et tous les autres animes
    cosine_similarities = linear_kernel(moyenne_vecteur, matrice).flatten()
        
    # Récupère les indices des animes les plus similaires
    similar_indices = cosine_similarities.argsort()[::-1][:50]
    
    # On ignore les 3 premiers (car ce sont souvent les mêmes animes déjà vus)
    candidats_indices = similar_indices[3:]
    
    # On mélange les indices pour plus de diversité
    indices_melanges = list(candidats_indices)
    np.random.shuffle(indices_melanges)
        
    # Récupère les détails des animes similaires, en excluant ceux déjà vus
    resultats = df.iloc[indices_melanges]
    recommandations = resultats[~resultats['Name'].isin(watched)]
        
    return recommandations
    
def recommander_par_top_score(df, n=1):
    # 1. On garde uniquement les lignes où le Score N'EST PAS 'Unknown'
    df_clean = df[df['Score'] != 'Unknown'].copy()
    
    # 2. On convertit la colonne en nombres (float)
    # C'est obligatoire, sinon le tri sera faux (ex: "9.1" > "10.0" en alphabétique)
    df_clean['Score'] = df_clean['Score'].astype(float)
    
    # 3. On trie par score décroissant
    top_animes = df_clean.sort_values(by='Score', ascending=False).head(n)
    
    return top_animes