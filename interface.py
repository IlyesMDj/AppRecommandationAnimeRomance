import streamlit as st
import pandas as pd
import requests 
import time
import recommender 


st.set_page_config(page_title="Anime Matcher", page_icon="💘", layout="wide")

# --- 1. FONCTIONS API (Image + Trailer) ---
@st.cache_data(ttl=3600)
def get_anime_info_api(anime_name, anime_id):
    """
    Récupère l'image et le trailer via l'API Jikan.
    Version sécurisée contre les erreurs.
    """
    # Info par défaut (image vide, pas de trailer)
    info = {
        "image": "https://via.placeholder.com/300x450?text=Image+Non+Dispo",
        "trailer": None,
        "url": f"https://myanimelist.net/anime/{anime_id}"
    }

    try:
        # 1. On force l'ID en entier (int) pour éviter les "4224.0" dans l'URL
        clean_id = int(anime_id)
        url = f"https://api.jikan.moe/v4/anime/{clean_id}"
        
        response = requests.get(url, timeout=5) # Timeout pour ne pas bloquer l'app
        
        # 2. On vérifie que la réponse est OK (Code 200)
        if response.status_code == 200:
            json_resp = response.json()
            
            # 3. Utilisation de .get() pour éviter les crashs si une clé manque
            data = json_resp.get('data', {})
            
            # Images
            images = data.get('images', {}).get('jpg', {})
            if images.get('large_image_url'):
                info['image'] = images['large_image_url']
            
            # Trailer
            trailer = data.get('trailer', {})
            if trailer.get('embed_url'):
                info['trailer'] = trailer['embed_url']
                
            # URL Officielle
            if data.get('url'):
                info['url'] = data['url']
                
        else:
            print(f"API Jikan Erreur {response.status_code} pour {anime_name}")

    except Exception as e:
        # Si ça plante (pas internet, erreur bizarre), on garde les infos par défaut
        print(f"Erreur technique API: {e}")
        
    return info
    
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
    
    # 0. Initialisation des listes si elles n'existent pas
    if 'queue' not in st.session_state:
        st.session_state.queue = []
    
    # On récupère les listes de l'utilisateur
    watched_list = st.session_state.user_data.get('watched', [])
    wishlist_list = st.session_state.user_data.get('wishlist', [])
    disliked_list = st.session_state.user_data.get('disliked', [])

    # Liste noire (ce qu'on ne veut plus voir)
    deja_vus = watched_list + disliked_list + wishlist_list

    # --- ÉTAPE 1 : REMPLIR LA FILE D'ATTENTE (QUEUE) SI ELLE EST VIDE ---
    if len(st.session_state.queue) == 0:
        
        recos = pd.DataFrame() # Conteneur vide pour les résultats
        
        # CAS 1 : L'utilisateur a les DEUX (Le top du top -> Hybride)
        if len(wishlist_list) > 0 and len(watched_list) > 0:
            recos = recommender.recommandation_hybride(wishlist_list, watched_list, matrice, df)
            # On stocke la source pour l'affichage du texte plus tard
            st.session_state.current_source = "hybride"

        # CAS 2 : L'utilisateur a SEULEMENT une Wishlist
        elif len(wishlist_list) > 0:
            recos = recommender.recommander_par_wishlist(wishlist_list, matrice, df)
            st.session_state.current_source = "wishlist"

        # CAS 3 : L'utilisateur a SEULEMENT des Déjà vus
        elif len(watched_list) > 0:
            recos = recommender.recommander_par_deja_vus(watched_list, matrice, df)
            st.session_state.current_source = "watched"

        # Traitement des résultats trouvés (si recos n'est pas vide)
        if not recos.empty:
            # On prend les noms
            nouveaux_candidats = recos['Name'].tolist()
            # On filtre ce qui est déjà connu
            candidats_propres = [a for a in nouveaux_candidats if a not in deja_vus]
            # On remplit la queue
            st.session_state.queue.extend(candidats_propres)

    # --- ÉTAPE 2 : RECUPÉRER LE PROCHAIN ANIME DE LA FILE ---
    if len(st.session_state.queue) > 0:
        nom_anime = st.session_state.queue[0]

        # Vérification de sécurité : si l'utilisateur vient de l'ajouter, on le saute
        if nom_anime in deja_vus:
            st.session_state.queue.pop(0)
            return get_next_anime() # Récursivité pour prendre le suivant

        # On récupère les infos de l'anime
        ligne_anime = df[df['Name'] == nom_anime]
        
        if not ligne_anime.empty:
            # On le retire de la file car on va l'afficher
            st.session_state.queue.pop(0)
            
            # --- Choix du texte explicatif selon la source ---
            source = st.session_state.get('current_source', 'unknown')
            raison = "🎯 Recommandation"
            
            if source == "hybride":
                raison = "⚡ Mix de vos goûts (Vus + Wishlist)"
            elif source == "wishlist":
                raison = "💖 Similaire à votre Wishlist"
            elif source == "watched":
                raison = "👀 Similaire à votre historique"

            return ligne_anime.iloc[0], raison

    # --- ÉTAPE 3 : PLAN DE SECOURS (COLD START) ---
    # Si la queue est toujours vide ici, c'est que l'utilisateur n'a rien, 
    # ou que les algos n'ont rien trouvé. On sort le Top Score.
    
    top_candidates = recommender.recommander_par_top_score(df, 150)
    
    for index, row in top_candidates.iterrows():
        if row['Name'] not in deja_vus:
            return row, "🏆 Populaire (Top Score Global)"

    # Si vraiment plus rien n'est dispo
    return None, None
        
# Initialisation
if st.session_state.current_anime is None:
    st.session_state.current_anime = get_next_anime()

# --- 5. ACTIONS ---
def action_utilisateur(action):
    # current_anime est maintenant un tuple (anime, raison)
    if st.session_state.current_anime[0] is not None:
        anime_data = st.session_state.current_anime[0]
        nom_anime = anime_data['Name']
        
        if action == 'like':
            st.session_state.user_data['wishlist'].append(nom_anime)
        elif action == 'dislike':
            st.session_state.user_data['disliked'].append(nom_anime)
        elif action == 'watched':
            st.session_state.user_data['watched'].append(nom_anime)
        
        # On vide la queue pour recalculer immédiatement avec les nouveaux goûts
        st.session_state.queue = [] 
        
    st.session_state.current_anime = get_next_anime()

def supprimer_item(liste_cible, nom_anime):
    """Permet de retirer un item d'une liste (Wishlist/Watched)"""
    if nom_anime in st.session_state.user_data[liste_cible]:
        st.session_state.user_data[liste_cible].remove(nom_anime)
        # On force le recalcul
        st.session_state.queue = []
        st.rerun()

# --- 6. INTERFACE GRAPHIQUE ---

# === SIDEBAR (STATS + GESTION) ===
with st.sidebar:
    st.header("📊 Tes Stats")
    st.write(f"💖 Wishlist: {len(st.session_state.user_data['wishlist'])}")
    st.write(f"👀 Vus: {len(st.session_state.user_data['watched'])}")
    
    st.divider()
    
    # --- WISHLIST AVEC SUPPRESSION ---
    st.subheader("💖 Ta Wishlist")
    if not st.session_state.user_data['wishlist']:
        st.caption("Vide pour l'instant.")
    
    for nom_anime in st.session_state.user_data['wishlist']:
        c1, c2 = st.columns([4, 1])
        
        with c1:
            # Lien cliquable
            ligne = df[df['Name'] == nom_anime]
            if not ligne.empty:
                mal_id = ligne.iloc[0]['MAL_ID']
                url = f"https://myanimelist.net/anime/{mal_id}"
                st.markdown(f"[{nom_anime}]({url})")
            else:
                st.write(f"{nom_anime}")
        
        with c2:
            # Bouton Poubelle (clé unique obligatoire)
            if st.button("🗑️", key=f"del_wish_{nom_anime}", help="Retirer de la liste"):
                supprimer_item('wishlist', nom_anime)

    st.divider()

    # --- DÉJÀ VUS AVEC SUPPRESSION ---
    st.subheader("👀 Déjà vus")
    if not st.session_state.user_data['watched']:
        st.caption("Vide pour l'instant.")

    for nom_anime in st.session_state.user_data['watched']:
        c1, c2 = st.columns([4, 1])
        with c1:
             st.caption(nom_anime)
        with c2:
             if st.button("🗑️", key=f"del_watch_{nom_anime}"):
                 supprimer_item('watched', nom_anime)

# === MAIN CONTENT ===
st.title("💘 Anime Matcher")

# Déballage du tuple (Anime, Raison)
anime, raison = st.session_state.current_anime

if anime is not None:
    # Appel API
    with st.spinner('Chargement des infos...'):
        infos_api = get_anime_info_api(anime['Name'], anime['MAL_ID'])

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(infos_api['image'], use_container_width=True)
        st.link_button("Voir sur MyAnimeList", infos_api['url'], use_container_width=True)
    
    with col2:
        # Affichage de la RAISON de la recommandation (Nouvelle fonctionnalité)
        if "Hybride" in raison:
            st.success(raison, icon="🎯")
        else:
            st.info(raison, icon="🏆")

        st.header(anime['Name'])
        
        # Affichage propre des genres (Tags)
        genres_list = anime['Genres'].split(', ')
        st.write(" ".join([f"`{g}`" for g in genres_list]))
        
        st.write(f"**Note :** ⭐ {anime['Score']}/10")
        
        with st.container(border=True):
            st.write(anime['sypnopsis'])

        # Trailer Youtube
        if infos_api['trailer']:
            with st.expander("🎬 Voir la bande-annonce (Trailer)", expanded=False):
                st.video(infos_api['trailer'])
        else:
            st.caption("🚫 Pas de trailer disponible via l'API.")
        
        st.divider()
        
        # Boutons d'action
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("👎 J'aime pas", use_container_width=True):
                action_utilisateur('dislike')
                st.rerun()
        with c2:
            if st.button("👀 Déjà vu", use_container_width=True):
                action_utilisateur('watched')
                st.rerun()
        with c3:
            if st.button("💖 Wishlist", type="primary", use_container_width=True):
                action_utilisateur('like')
                st.rerun()

else:
    st.balloons()
    st.success("🎉 Incroyable ! Tu as fait le tour de tout le catalogue !")
    if st.button("Tout recommencer (Reset)"):
        st.session_state.clear()
        st.rerun()