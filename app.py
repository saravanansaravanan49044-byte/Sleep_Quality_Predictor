from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

saved_data = joblib.load("models/sleep_model.pkl")
model = saved_data["model"]
features = saved_data["features"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_data = {
            feature: data.get(feature)
            for feature in features
        }

        input_df = pd.DataFrame([input_data], columns=features)
        prediction = model.predict(input_df)[0]

        return jsonify({
            "prediction": str(prediction),
            "message": "Sleep quality prediction completed!"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)