from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.utils.extmath import softmax

app = Flask(__name__)

# Load model
model = joblib.load("svm_sentiment_pipeline.pkl")

# =========================
# ROUTES
# =========================

# DEFAULT: LANGSUNG KE PREDIKSI
@app.route("/")
def home():
    return render_template("predict.html")

# VISUALIZATION TAB
@app.route("/visualization")
def visualization():
    return render_template("index.html")

# PREDIKSI SENTIMEN
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    text_input = ""

    if request.method == "POST":
        text_input = request.form["text"]
        prediction = model.predict([text_input])[0]

        decision = model.decision_function([text_input])
        probs = softmax(decision)
        confidence = float(np.max(probs))

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence,
        text_input=text_input
    )

if __name__ == "__main__":
    app.run(debug=True)
