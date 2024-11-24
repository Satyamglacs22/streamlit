import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
data = pd.read_csv("C:/Users/sk725/Downloads/diabetes.csv")


# Split the data into features and target variable
X = data.drop(columns='Outcome')
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Save the model using pickle
pickle.dump(model, open('diabetes_model.sv', 'wb'))

