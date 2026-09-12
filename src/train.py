import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

# Constants
DATA_PATH = "../data/BankChurners.csv"  
MODELS_DIR = "../models"

def load_data(path):
    print("Loading data...")
    df = pd.read_csv(path)
    # Binarize target: 1 for Churn, 0 for Retained
    df['Attrition_Flag'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
    X = df.drop(columns=['CLIENTNUM', 'Attrition_Flag'])
    y = df['Attrition_Flag']
    return train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

def build_preprocessor():
    print("Building preprocessing pipeline...")
    edu_cat = [["Uneducated", "Unknown", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"]]
    income_cat = [["Unknown", "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"]]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("OHE", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), 
             ["Gender", "Marital_Status", "Card_Category"]),
            ("scale", StandardScaler(), 
             ["Customer_Age", "Dependent_count", "Months_on_book", "Total_Relationship_Count",
              "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
              "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
              "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"]),
            ("edu_ord", OrdinalEncoder(categories=edu_cat), ["Education_Level"]),
            ("inc_ord", OrdinalEncoder(categories=income_cat), ["Income_Category"]),
        ],
        remainder="passthrough"
    )
    return preprocessor

def train_champion_model(X_train, y_train, preprocessor):
    print("Training XGBoost Champion Model...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Calculate class weight for imbalance
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=42
    )
    model.fit(X_train_processed, y_train)
    return model, preprocessor

def save_artifacts(model, preprocessor):
    print("Saving artifacts for deployment...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    model.save_model(os.path.join(MODELS_DIR, "xgb_churn_model.json"))
    print(f"Artifacts successfully saved to {MODELS_DIR}/")

def main():
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    
    # 2. Build Pipeline
    preprocessor = build_preprocessor()
    
    # 3. Train Model
    model, fitted_preprocessor = train_champion_model(X_train, y_train, preprocessor)
    
    # 4. Export Artifacts
    save_artifacts(model, fitted_preprocessor)

if __name__ == "__main__":
    main()