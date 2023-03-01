from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vect.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ["POST"])
def predict():
    if request.method == "POST":
        text_features = request.form["title_maker"]
        new_title = vectorizer.transform([text_features.lower()])
        prediction = model.predict(new_title)[0]
        return render_template("index.html", prediction_text = "Przewidywana liczba kliknięć w link, to: {}".format(prediction))

if __name__ == '__main__':
    app.run(port=8000,debug=True)
