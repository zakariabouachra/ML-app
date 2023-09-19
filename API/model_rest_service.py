from flask import Flask, request, jsonify
import joblib 
import pandas as pd

# Créer l'application Flask
app = Flask(__name__)

def load_model():
    model = joblib.load('Classification/pima-indians-diabetes_model.pickle')
    return model


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    input_data = pd.DataFrame.from_records(input_data).values
    model = load_model()
    if model is None:
        return jsonify({'error': 'Invalid prediction type'})
    else:
        # Faire la prédiction avec le modèle chargé
        prediction = model.predict(input_data)
        response = {'prediction': float(prediction[0])}
       

        return jsonify(response)



if __name__ == '__main__':
    app.run(port=8806, debug=True)
