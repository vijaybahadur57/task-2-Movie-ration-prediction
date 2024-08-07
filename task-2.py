import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('movies.csv')

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing ratings
data.dropna(subset=['Rating'], inplace=True)

# Fill missing values in other columns if necessary
data['Genre'].fillna('Unknown', inplace=True)
data['Director'].fillna('Unknown', inplace=True)
data['Actors'].fillna('Unknown', inplace=True)

# Convert categorical data to numerical data
data = pd.get_dummies(data, columns=['Genre', 'Director', 'Actors'], drop_first=True)

# Define features and target variable
X = data.drop('Rating', axis=1)
y = data['Rating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')