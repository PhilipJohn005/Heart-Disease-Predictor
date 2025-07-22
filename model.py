import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def load_data(path='./data/heart_disease_uci.csv'):
    df = pd.read_csv(path)
    df = df.dropna()
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(['id', 'num'], axis=1, errors='ignore')
    df = pd.get_dummies(df, drop_first=True)
    return df

def train_model():
    print("Loading Data....")
    df = load_data()

    X = df.drop('target', axis=1)
    y = df['target']

    print("Train Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=52)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    os.makedirs('models', exist_ok=True)

    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('models/features.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    print("✅ Model and features saved successfully.")

if __name__ == "__main__":
    train_model()
