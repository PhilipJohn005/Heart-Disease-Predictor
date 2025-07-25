import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pickle


# Load & preprocess data
def load_data(path='./Data/heart_disease_uci.csv'):
    
    df = pd.read_csv(path)
    df = df.dropna()
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(['id', 'num'], axis=1, errors='ignore')
    df = pd.get_dummies(df, drop_first=True)
    return df

def compare_models():
    print("Loading dataset......")
    df = load_data()
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=52),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Gradient_Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = []
    print("Training....")

    for name, model in models.items():
        print(f"Training model {name}.....")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 3),
            "F1 Score": round(f1, 3),
            "ROC AUC": round(auc, 3)
        })
        
        with open(f"models/{name}.pkl","wb") as f:
            pickle.dump(model,f)
            
            
    print("Finished Training........")  
    
    df_results = pd.DataFrame(results).sort_values(by="ROC AUC", ascending=False)
    print("\n🔍 Model Comparison:\n")
    print(df_results)

if __name__ == "__main__":
    compare_models()
