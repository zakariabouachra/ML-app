import streamlit as st
from streamlit import session_state as state
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, Binarizer
import requests
import subprocess
import pickle


def model_Classification(model_type):
    if model_type == "Régression logistique":
        return LogisticRegression()
    elif model_type == "Arbres de décision":
        return DecisionTreeClassifier()
    elif model_type == "KNeighborsClassifier":
        return KNeighborsClassifier()
    elif model_type == "GaussianNB":
        return GaussianNB()
    elif model_type == "Support Vector Machines":
        return SVC()
def transform_model(transform):
    if transform == "Standard Scaler":
        return MinMaxScaler()
    elif transform == "Normalization":
        return Normalizer()
    elif transform == "Standarization":
        return StandardScaler()
    elif transform == "Binarization":
        return Binarizer()
def model_regression(model_type):
    if model_type == "Linear Regression":
        return LinearRegression()
    elif model_type == "Ridge":
        return Ridge()
    elif model_type == "Lasso":
        return Lasso()
    elif model_type == "ElasticNet":
        return ElasticNet()
    elif model_type == "DecisionTreeRegressor":
        return DecisionTreeRegressor()
    elif model_type == "KNeighborsRegressor":
        return KNeighborsRegressor()
    elif model_type == "SVR":
        return SVR()
def saveEntrainementModel(fichiername, type, model, transformation):
    if type == 'Classification':
        filename = f'API/Classification/{fichiername}_model.pickle'
    else:
        filename = f'API/Regression/{fichiername}_model.pickle'

    data = {'model': model}

    if transformation is not None:
        data['transformation'] = transformation

    with open(filename, 'wb') as file:
        pickle.dump(data, file)
def apprentissageEtanalyse():
    if 'results' not in state:
        state.results = None

    # Afficher les données
    st.subheader("Aperçu du dataset")
    st.write(dataset.head())

    array = dataset.values
    # get inputs (all variables except the class)
    X = array[:, 0:-1]
    if prediction_type == "Regression":
        Y = array[:, -1].astype(int)  # Convert the target variable to integer for classification
    else:
        Y = array[:, -1]  # For regression, keep the target variable as it is (continuous)

    transforms = [None, "Standard Scaler", "Normalization", "Standarization", "Binarization"]
    algorithms_classification = ["Régression logistique", "Arbres de décision", "KNeighborsClassifier", "GaussianNB", "Support Vector Machines"]
    algorithms_regression = ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "DecisionTreeRegressor", "KNeighborsRegressor", "SVR"]

    results = []
    
    with st.spinner("Veuillez patienter S.V.P ..."):
        for transform in transforms:
            transformer = transform_model(transform)
            if transformer is not None:
                rescaled_X = transformer.fit_transform(X)
            else:
                rescaled_X = X

            # Boucle pour les algorithmes de classification
            if prediction_type == "Classification":
                for algorithm in algorithms_classification:
                    # Prepare models for classification
                    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
                    model = model_Classification(algorithm)
                    # Métriques d'évaluation
                    model.fit(rescaled_X, Y)
                    accuracy = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring='accuracy').mean()
                    precision = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring='precision').mean()
                    recall = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring='recall').mean()
                    f1 = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring='f1').mean()
                    roc_auc = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring='roc_auc').mean()

                    results.append({
                        'algorithm': algorithm,
                        'transformation': transform,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'roc_auc': roc_auc,
                    })

            # Boucle pour les algorithmes de régression
            if prediction_type == "Regression":
                for algorithm in algorithms_regression:
                    # Prepare models for regression
                    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
                    model = model_regression(algorithm)
                    model.fit(rescaled_X, Y)
                    # Métriques d'évaluation
                    neg_mean_absolute_error = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring='neg_mean_absolute_error').mean()
                    neg_mean_squared_error = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring='neg_mean_squared_error').mean()
                    r2 = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring='r2').mean()

                    results.append({
                        'algorithm': algorithm,
                        'transformation': transform,
                        'neg_mean_absolute_error': neg_mean_absolute_error,
                        'neg_mean_squared_error': neg_mean_squared_error,
                        'r2': r2
                    })
            
        state.results = results
def main():
    best_algorithm = None
    best_transformation = None
    best_metric_score = 0.0
    best_metric = None

    if 'best_algorithm' not in state:
        state.best_algorithm = None

    if 'best_transformation' not in state:
        state.best_transformation = None
    # Trouver le meilleur algorithme et transformation basé sur la métrique sélectionnée
    best_metric = 'accuracy' if prediction_type == 'Classification' else 'r2'
    for result in state.results:
        if best_metric in result and result[best_metric] > best_metric_score:
            best_metric_score = result[best_metric]
            best_algorithm = result['algorithm']
            best_transformation = result['transformation']
       
    # Store the best algorithm and transformation in the session state
    state.best_algorithm = best_algorithm
    state.best_transformation = best_transformation

    # Afficher le meilleur algorithme et la meilleure transformation
    st.subheader("Meilleur algorithme et transformation")
    st.write("Algorithme:", best_algorithm)
    st.write("Transformation:", best_transformation)
   
    with st.expander("Analyse et explication"):
        display_analysis_and_explanation(best_algorithm, best_transformation, state.results,prediction_type) 
    with st.expander("Résultats des autres algorithmes et transformations"):
        display_other_algorithm_results(state.results,best_algorithm, best_transformation,prediction_type) 

    st.markdown("Veux-tu faire des prédictions ? [Clique ici](http://localhost:8501/)")
def prediction_tempsreel():
    # Prédiction en temps réel
    st.subheader("Prédiction en temps réel")

    # Créer des sliders pour saisir les valeurs en temps réel
    slider_values = []

    for i in range(len(dataset.columns) - 1):
        value = st.slider(f"Saisir la valeur pour {dataset.columns[i]}", float(dataset[dataset.columns[i]].min()), float(dataset[dataset.columns[i]].max()),float(dataset[dataset.columns[i]].mean()))
        slider_values.append(value)
    input_data = pd.DataFrame([slider_values], columns=dataset.columns[:-1])

    # Préparer les données pour l'envoi
    input_data_json = input_data.to_json(orient='records')

    # Envoyer les données au serveur
    response = requests.post('http://localhost:8806/predict', json=input_data_json)
    
    if response.status_code == 200:
        prediction_result = response.json()

        if prediction_type == "Classification":
            prediction = prediction_result['prediction'][0]
            probabilty = prediction_result['probability'][0][int(prediction)]
            
            if prediction == 0:
                st.write("Prédiction: Negative-Diabete")
            else:
                st.write("Prédiction: Positive-Diabete")
            st.write("Probabilty: ", probabilty)
        else:
            prediction = prediction_result['prediction'][0]
            st.write("Prédiction:", prediction)
    else:
        st.write("Erreur lors de la prédiction")
def run_another_app():
    #subprocess.Popen(["python", "API/model_rest_service.py"])
    subprocess.Popen(["streamlit", "run", "API/model_rest_client.py"])
def show_dialog_box():
    st.subheader("Sélectionner les options:")
    delimiter_choices = [",", ";", "white_Space"]
    prediction_choices = ["Classification", "Regression"]
    delimiter = st.selectbox("Délimiteur", delimiter_choices)
    prediction_type = st.selectbox("Type de prédiction", prediction_choices)
    go_button = st.button("Go Baby")
    
    return delimiter, prediction_type, go_button
def display_analysis_and_explanation(best_algorithm, best_transformation, results, prediction_type):
    st.write("Le meilleur algorithme sélectionné est :", best_algorithm)
    st.write("La meilleure transformation sélectionnée est :", best_transformation)

    st.write("Explication :")
    st.write("Pour choisir le meilleur algorithme et la meilleure transformation, nous avons effectué une analyse comparative des performances de différents algorithmes et transformations sur les données du dataset. Nous avons utilisé la validation croisée (cross-validation) avec 10 plis (folds) pour évaluer chaque combinaison d'algorithme et de transformation.")

    if prediction_type == 'Classification':
        st.write("Pour les tâches de classification, nous avons évalué chaque algorithme en utilisant les métriques suivantes :")
        st.write("- Accuracy (précision globale)")
        st.write("- Precision (précision)")
        st.write("- Recall (rappel)")
        st.write("- F1-score (score F1)")
        st.write("- ROC AUC (Aire sous la courbe ROC)")

        # Afficher les scores de tous les metrics pour le meilleur algorithme
        st.subheader("Scores des métriques pour le meilleur algorithme de classification")
        for result in results:
            if result['algorithm'] == best_algorithm and result['transformation'] == best_transformation:
                st.write("Algorithm:", result['algorithm'])
                st.write("Transformation:", result['transformation'])
                st.write("Accuracy:", f"{result['accuracy']:.2f}")
                st.write("Precision:", f"{result['precision']:.2f}")
                st.write("Recall:", f"{result['recall']:.2f}")
                st.write("F1-score:", f"{result['f1']:.2f}")
                st.write("ROC AUC:", f"{result['roc_auc']:.2f}")

    elif prediction_type == 'Regression':
        st.write("Pour les tâches de régression, nous avons évalué chaque algorithme en utilisant les métriques suivantes :")
        st.write("- Negative Mean Absolute Error (Erreur absolue moyenne négative)")
        st.write("- Negative Mean Squared Error (Erreur quadratique moyenne négative)")
        st.write("- R-squared (Coefficient de détermination)")

        # Afficher les scores de tous les metrics pour le meilleur algorithme
        st.subheader("Scores des métriques pour le meilleur algorithme de régression")
        for result in results:
            if result['algorithm'] == best_algorithm and result['transformation'] == best_transformation:
                st.write("Algorithm:", result['algorithm'])
                st.write("Transformation:", result['transformation'])
                st.write("Neg Mean Absolute Error:", f"{result['neg_mean_absolute_error']:.2f}")
                st.write("Neg Mean Squared Error:", f"{result['neg_mean_squared_error']:.2f}")
                st.write("R-squared:", f"{result['r2']:.2f}")

    st.write("Après avoir calculé les scores de chaque métrique pour chaque combinaison, nous avons sélectionné l'algorithme et la transformation qui ont obtenu le meilleur score pour l'une des métriques pertinentes en fonction du type de prédiction (classification ou régression). Dans ce cas, nous avons choisi la métrique 'accuracy' pour les tâches de classification et 'r2' pour les tâches de régression.")

    st.write("Il est important de noter que le choix du meilleur algorithme et de la meilleure transformation dépend des caractéristiques du dataset et des objectifs du modèle. Ces résultats peuvent varier en fonction des données spécifiques et des nouvelles observations.")

    st.write("Nous espérons que cette analyse et cette explication vous permettent de mieux comprendre le processus de sélection du meilleur algorithme et de la meilleure transformation pour ce projet.")
def display_other_algorithm_results(results, best_algorithm, best_transformation, prediction_type):
    
    if prediction_type == 'Classification':
        for result in results:
            if result['algorithm'] != best_algorithm or result['transformation'] != best_transformation:
                st.write("Algorithm:", result['algorithm'])
                st.write("Transformation:", result['transformation'])
                st.write("Accuracy:", f"{result['accuracy']:.2f}")
                st.write("Precision:", f"{result['precision']:.2f}")
                st.write("Recall:", f"{result['recall']:.2f}")
                st.write("F1-score:", f"{result['f1']:.2f}")
                st.write("ROC AUC:", f"{result['roc_auc']:.2f}")
                st.write("")

    elif prediction_type == 'Regression':
        for result in results:
            if result['algorithm'] != best_algorithm or result['transformation'] != best_transformation:
                st.write("Algorithm:", result['algorithm'])
                st.write("Transformation:", result['transformation'])
                st.write("Neg Mean Absolute Error:", f"{result['neg_mean_absolute_error']:.2f}")
                st.write("Neg Mean Squared Error:", f"{result['neg_mean_squared_error']:.2f}")
                st.write("R-squared:", f"{result['r2']:.2f}")
                st.write("")

    st.write("Ces résultats vous permettent de comparer les performances des autres algorithmes et transformations évalués par rapport au meilleur algorithme et à la meilleure transformation sélectionnés précédemment.")

uploaded_file = st.sidebar.file_uploader('Insérer votre dataset')

if uploaded_file is not None:
    delimiter, prediction_type, go = show_dialog_box()

    if go:
        if delimiter == "white_Space":
            upload_dataset = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
        else:
            upload_dataset = pd.read_csv(uploaded_file, delimiter=delimiter, header=None)

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

        apprentissageEtanalyse()
        main()
        saveEntrainementModel(uploaded_file.name,prediction_type,state.best_algorithm, state.best_transformation)


        
    