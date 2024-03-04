# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the wine quality dataset
wine_data = pd.read_csv('winequality-red.csv')

# Data preprocessing
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Calculating and printing the accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Evaluating the model
print("Classification Report:\n", classification_report(y_test, predictions))
