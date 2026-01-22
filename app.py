from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open("model/titanic_survival_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        pclass = int(request.form["pclass"])
        sex = int(request.form["sex"])
        age = float(request.form["age"])
        fare = float(request.form["fare"])
        embarked = int(request.form["embarked"])

        input_df = pd.DataFrame(
            [[pclass, sex, age, fare, embarked]],
            columns=FEATURES
        )

        input_scaled = scaler.transform(input_df)
        result = model.predict(input_scaled)[0]

        prediction = "Survived" if result == 1 else "Did Not Survive"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
