import pandas as pd

df = pd.read_csv('animes.csv')

# On supprime les lignes qui n'ont PAS de genre OU pas de sysnopsis
df.dropna(subset=['Genres', 'sypnopsis'], inplace=True)

# On supprime les lignes dont le synopsis est une phrase par défaut
df = df[~df['sypnopsis'].str.contains("No synopsis information has been added to this title. Help improve our database by adding a synopsis here .", na=False)]

# On crée un filtre pour les animes de genre Romance
filtreRomance = df['Genres'].str.contains('Romance', na=False, case=False)

# On applique ce filtre pour créer un nouveau tableau
df_romance = df[filtreRomance]

# drop=True évite que l'ancien numéro devienne une nouvelle colonne
df_romance = df_romance.reset_index(drop=True)

# On sauvegarde le nouveau tableau dans un fichier CSV
df_romance.to_csv('anime_romance.csv', index=False)

# Affichage du nombre d'animes de genre Romance
print(f"Nombre d'animes de genre Romance : {len(df_romance)}")