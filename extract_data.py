import pandas as pd

TXT_FOLDER = 'Data/Text/'
CSV_FOLDER = 'Data/Csv/'

# Charger le fichier CSV
df = pd.read_csv('grading_dataset.csv')

# Parcourir les différents topics
for topic in df['Topic'].unique():
    # Filtrer le DataFrame par topic
    topic_df = df[df['Topic'] == topic]

    # Trier le DataFrame par "Assigned Grade" de manière décroissante
    topic_df = topic_df.sort_values(by='Assigned Grade', ascending=False)

    # Renommer les colonnes
    topic_df = topic_df.rename(columns={'Review Text': 'review', 'Assigned Grade': 'label'})

    # Exclure la colonne 'Topic'
    topic_df = topic_df[['review', 'label']]

    # Sauvegarder le DataFrame dans un nouveau fichier CSV
    topic_df.to_csv(CSV_FOLDER + str(topic) + '.csv', index=False)
