# ML-app
##Introduction
Cette application ML est conçue pour fournir des services de classification et de régression via une API REST. Elle permet aux utilisateurs de faire des prédictions en envoyant des données à travers des requêtes HTTP.

##Configuration requise

Python 3.8 ou supérieur
Flask
NumPy
Pandas
scikit-learn

##Installation

Pour installer et configurer l'application, suivez ces étapes :

```
git clone https://github.com/zakariabouachra/ML-app.git
cd ML-app
pip install -r requirements.txt
```

##Utilisation

Pour démarrer le serveur REST, exécutez :
```
python API/model_rest_service.py
```

Pour faire une requête au serveur, utilisez le script client comme suit :

```
streamlit run API/model_rest_client.py
```

##Contribuer
Les contributions à ce projet sont les bienvenues. Pour contribuer, veuillez forker le dépôt, créer une branche de fonctionnalité, et soumettre une pull request.
