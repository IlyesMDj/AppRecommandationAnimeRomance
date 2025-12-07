import streamlit as st
import pandas as pd
import requests 
import time
import recommender 


st.set_page_config(page_title="Anime Matcher", page_icon="💘", layout="wide")

@st.cache_data(ttl=3600) # On garde l'image en mémoire 1h pour ne pas spammer l'API
def get_anime_image(anime_name):
    """
    Cherche l'image de l'animé via l'API Jikan.
    """
    url = f"https://api.jikan.moe/v4/anime?q={anime_name}&limit=1"
    try:
        response = requests.get(url)
        data = response.json()
        
        # On va chercher l'image en haute qualité si possible
        if data['data']:
            image_url = data['data'][0]['images']['jpg']['large_image_url']
            return image_url
        else:
            return "https://via.placeholder.com/300x450?text=Image+Not+Found"
    except:
        return "https://via.placeholder.com/300x450?text=API+Error"
    
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'wishlist': [],
        'watched': [],
        'disliked': []
    }
    
# On stocke l'anime qu'on est en train de regarder
if 'current_anime' not in st.session_state:
    st.session_state.current_anime = None

# Une file d'attente pour les recommandations
if 'queue' not in st.session_state:
    st.session_state.queue = []

#² Chargement des données et préparation du modèle
df = recommender.charger_donnees()
matrice, indices = recommender.preparer_modele(df)

# Fonction pour obtenir le prochain anime à recommander
def get_next_anime():
    # 0. On prépare la liste noire
    deja_vus = st.session_state.user_data['watched'] + \
               st.session_state.user_data['disliked'] + \
               st.session_state.user_data['wishlist']

    # --- STRATÉGIE 1 : TU AS DÉJÀ DES GOÛTS (Wishlist non vide) ---
    if len(st.session_state.user_data['wishlist']) > 0:
        
        # On récupère TOUT le catalogue classé par ressemblance avec tes likes
        toutes_recos = recommender.recommander_par_wishlist(
            st.session_state.user_data['wishlist'], 
            matrice, 
            df
        )
        for index, row in toutes_recos.iterrows():
            if row['Name'] not in deja_vus:
                return row

    # --- STRATÉGIE 2 : EXPLORATION (Si tu as juste des 'Vus' sans 'Likes') ---
    elif len(st.session_state.user_data['watched']) > 0:
        recos_exploration = recommender.recommander_exploration(
             st.session_state.user_data['watched'],
             matrice,
             df
        )
        for index, row in recos_exploration.iterrows():
            if row['Name'] not in deja_vus:
                return row

    # --- STRATÉGIE 3 : DÉMARRAGE À FROID (Top Score) ---
    # On prend TOUT le tableau trié par note
    top_candidates = recommender.recommander_par_top_score(df, n=len(df))
    
    
    #On parcourt jusqu'à trouver un anime non vu
    for index, row in top_candidates.iterrows():
        if row['Name'] not in deja_vus:
            return row
            
    # Si vraiment on a tout vu
    return None
    
    
if st.session_state.current_anime is None:
    st.session_state.current_anime = get_next_anime()
    
def action_utilisateur(action):
    if st.session_state.current_anime is not None:
        nom_anime = st.session_state.current_anime['Name']
        
        if action == 'like':
            st.session_state.user_data['wishlist'].append(nom_anime)
        elif action == 'dislike':
            st.session_state.user_data['disliked'].append(nom_anime)
        elif action == 'watched':
            st.session_state.user_data['watched'].append(nom_anime)
        
    st.session_state.current_anime = get_next_anime()
    
with st.sidebar: 
    st.header("📊 Tes Stats")
    st.write(f"💖 Wishlist: {len(st.session_state.user_data['wishlist'])}")
    st.write(f"👀 Vus: {len(st.session_state.user_data['watched'])}")
    st.write(f"👎 Passés: {len(st.session_state.user_data['disliked'])}")
    
    st.divider()
    st.subheader("Ta Wishlist")
    
    for anime in st.session_state.user_data['wishlist']:
        st.caption(f"- {anime}")
        
st.title("💘 Anime Matcher")

anime = st.session_state.current_anime

if anime is not None:
    # Récupération de l'image (peut prendre 1 seconde)
    with st.spinner('Recherche de l\'image...'):
        img_url = get_anime_image(anime['Name'])

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(img_url, use_container_width=True)
    
    with col2:
        st.header(anime['Name'])
        st.markdown(f"**Note :** ⭐ {anime['Score']} | **Genres :** {anime['Genres']}")
        st.info(anime['sypnopsis'])
        
        st.divider()
        
        # Les 3 Boutons
        b1, b2, b3 = st.columns(3)
        
        with b1:
            if st.button("👎 Je n'aime pas", use_container_width=True):
                action_utilisateur('dislike')
                st.rerun()
        
        with b2:
            if st.button("👀 Déjà vu", use_container_width=True):
                action_utilisateur('watched')
                st.rerun()

        with b3:
            if st.button("💖 J'aime (Wishlist)", type="primary", use_container_width=True):
                action_utilisateur('like')
                st.rerun()

else:
    st.success("🎉 Incroyable ! Tu as fait le tour de tout le catalogue !")
    if st.button("Recommencer à zéro"):
        st.session_state.clear()
        st.rerun()