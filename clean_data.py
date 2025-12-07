import pandas as pd

# 1. Chargement des données
df = pd.read_csv('animes.csv')

# 2. Nettoyage initial (Vides et Synopsis par défaut)
df.dropna(subset=['Genres', 'sypnopsis'], inplace=True)
df = df[~df['sypnopsis'].str.contains("No synopsis information", na=False, regex=False)]

# 3. Définition des filtres
genres_interdits = ['Horror', 'Demons', 'Hentai', 'Gore', 'Ecchi', 'Military']
mots_bannis = [
    '1st Season', 'Season 2', 'Season 3', 'Season 4', '2nd Season', '3rd Season', 
    'Part 2', 'Part II', 'Second Season', 'Third Season',
    'OVA', 'Ova', 'Special', 'Specials', 'Movie', 'Music', 'Live', 
    r'\(TV\)' # Le 'r' et les '\' sont importants pour cibler les parenthèses
]

# 4. Application des filtres (Genre)
# On combine les filtres : Doit contenir Romance ET ne pas contenir les interdits
mask_romance = df['Genres'].str.contains('Romance', na=False, case=False)
mask_pas_interdits = ~df['Genres'].str.contains('|'.join(genres_interdits), na=False, case=False, regex=True)

# On crée le DataFrame propre avec .copy() pour éviter les warnings
df_romance = df[mask_romance & mask_pas_interdits].copy()

# 5. Application des filtres (Titres "Suites")
mask_titre_propre = ~df_romance['Name'].str.contains('|'.join(mots_bannis), case=False, na=False, regex=True)
df_romance = df_romance[mask_titre_propre]

# 6. Suppression des doublons (Synopsis)
df_romance = df_romance.drop_duplicates(subset=['sypnopsis'], keep='first')

# 7. Suppression des doublons (Titre court)
# On coupe aux ":" (ex: "Clannad: After Story" -> "Clannad")
df_romance['titre_court'] = df_romance['Name'].str.split(':', regex=False).str[0]
# On coupe aux " (" (ex: "Fullmetal Alchemist (2009)" -> "Fullmetal Alchemist")
df_romance['titre_court'] = df_romance['titre_court'].str.split(' (', regex=False).str[0]

df_romance = df_romance.drop_duplicates(subset=['titre_court'], keep='first')
df_romance = df_romance.drop(columns=['titre_court'])

# 8. Finalisation et Sauvegarde
df_romance = df_romance.reset_index(drop=True)
df_romance.to_csv('anime_romance.csv', index=False)

print(f"Nombre d'animes de genre Romance : {len(df_romance)}")