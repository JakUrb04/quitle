import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Wczytywanie danych
data = pd.read_csv("new_data.csv")

# Dzielenie danych na zbiór treningowy i testowy
train_data, test_data, train_labels, test_labels = train_test_split(data["Tytuł"], data["Kliknięcia linku"], test_size=0.2, random_state=0)

# Tokenizacja i kodowanie tytułów artykułów
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)

# Trening modelu
model = LogisticRegression()
model.fit(train_features, train_labels)
model.predict(train_features)
model.predict(test_features)

pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vect.pkl","wb"))

# Ocena modelu
#train_pred = model.predict(train_features)
#test_pred = model.predict(test_features)

# Wykorzystanie modelu
#new_title = str(input("Wprowadź nowy tytuł: "))
#new_title_features = vectorizer.transform([new_title])
#new_title_click = model.predict(new_title_features)

#print(f"Przewidywana ilość kliknięć dla tytułu: {new_title}, to {new_title_click}")