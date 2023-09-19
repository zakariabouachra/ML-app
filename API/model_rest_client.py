import streamlit as st
import pandas as pd
import requests
import json
from flask import Flask, request
import pandas as pd

def prediction_tempsreel(dataset):
    # Prédiction en temps réel
    st.subheader("Prédiction en temps réel")

    # Créer des sliders pour saisir les valeurs en temps réel
    slider_values = []

    for i in range(len(dataset.columns) - 1):
        value = st.slider(f"Saisir la valeur pour {dataset.columns[i]}", float(dataset[dataset.columns[i]].min()), float(dataset[dataset.columns[i]].max()), float(dataset[dataset.columns[i]].mean()))
        slider_values.append(value)
    input_data = pd.DataFrame([slider_values], columns=dataset.columns[:-1])

    # Préparer les données pour l'envoi
    input_data_json = input_data.to_json(orient='records')
    


    # Envoyer les données au serveur
    response = requests.post('http://localhost:8806/predict', json=json.loads(input_data_json), headers={'Content-Type': 'application/json'})


    if response.status_code == 200:
        prediction_result = response.json()     
        prediction = prediction_result['prediction']
        st.write("Prédiction:", prediction)
    else:
        st.write("Erreur lors de la prédiction")


uploaded_file = st.sidebar.file_uploader('Insérer votre dataset')

if uploaded_file is not None:
   
    delimiter_choices = st.selectbox("selectionner le delimiteur",[",", ";", "white_Space"])
    if delimiter_choices == "white_Space":
        upload_dataset = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
    else:
        upload_dataset = pd.read_csv(uploaded_file, delimiter=delimiter_choices, header=None)

   # Vérifier si la première ligne ressemble à un en-tête
    first_row = upload_dataset.iloc[0]
    is_header = first_row.apply(lambda x: isinstance(x, str)).all()

    if is_header:
        # La première ligne est un en-tête, définir les colonnes
        upload_dataset.columns = first_row

        # Exclure la première ligne du dataset
        dataset = upload_dataset[1:]
    else:
        # La première ligne n'est pas un en-tête
        dataset = upload_dataset

    dataset = upload_dataset
    prediction_tempsreel(dataset)
    
 
