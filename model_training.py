import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Read the data file (Using data from the Data Professor)
df = pd.read_csv("./dataset/penguins_cleaned.csv")

# Define features and targets
X_raw = df.drop("species", axis=1)
print(X_raw)
y_raw = df.species
print(y_raw)

# Data encoding
encode = ["island", "sex"]
X_encoded = pd.get_dummies(X_raw, columns=encode)

target_mapper = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
y_encoded = y_raw.apply(lambda x: target_mapper[x])

# Split data for cross-validation
x_train, x_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=1
)

# Train the ML model
clf = RandomForestClassifier()
clf.fit(x_train, y_train)

# Display the accuracy of the model
print(f"Train accuracy: {accuracy_score(y_train, clf.predict(x_train))}")
print(f"Test accuracy: {accuracy_score(y_test, clf.predict(x_test))}")

# Save the model
joblib.dump(clf, "penguin_classifier_model.pkl")

print("Model creation completed!")
