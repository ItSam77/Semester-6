import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib


def load_and_prepare_data(file_path='processed_data.csv'):
    """Load and prepare the dataset for modeling"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # Display basic info about the dataset
    print("\nDataset Info:")
    print(df.info())
    
    # Check class distribution
    print("\nClass distribution:")
    print(df['Attrition'].value_counts())
    print(f"Class balance ratio: {df['Attrition'].value_counts(normalize=True)}")
    
    # Separate features and target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    return X, y

def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance"""
    print("\nApplying SMOTE...")
    print(f"Before SMOTE - Class distribution: {np.bincount(y_train)}")
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Class distribution: {np.bincount(y_train_smote)}")
    print(f"Training set shape after SMOTE: {X_train_smote.shape}")
    
    return X_train_smote, y_train_smote

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("\nTraining Random Forest model...")
    
    # Initialize Random Forest with good default parameters
    rf_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        max_features='sqrt',
        bootstrap=False
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    print("Model training completed!")
    
    return rf_model

def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate the model and generate classification report"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    
    return y_pred, y_pred_proba

def perform_cross_validation(model, X, y, cv_folds=5):
    """Perform k-fold cross validation"""
    print("\n" + "="*50)
    print("K-FOLD CROSS VALIDATION")
    print("="*50)
    
    # Use StratifiedKFold to maintain class balance in each fold
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Perform cross validation
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    
    print(f"Cross-validation with {cv_folds} folds:")
    print(f"Accuracy scores for each fold: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Standard Deviation: {cv_scores.std():.4f}")
    
    return cv_scores

def save_model(model, filename):
    """Save the trained model"""
    joblib.dump(model, filename)
    print(f"Model saved as: {filename}")

def main():
    """Main function to run the complete modeling pipeline"""
    print("EMPLOYEE ATTRITION PREDICTION - RANDOM FOREST MODEL")
    print("="*60)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Split the data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Apply SMOTE to training data only
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    
    # Train Random Forest model (NO STANDARD SCALING as requested)
    rf_model = train_random_forest(X_train_smote, y_train_smote)
    
    # Evaluate the model
    y_pred, y_pred_proba = evaluate_model(
        rf_model, X_test, y_test, X.columns
    )
    
    # Perform cross validation on original data (before SMOTE)
    # Note: We use original data for CV to get a more realistic estimate
    perform_cross_validation(rf_model, X, y, cv_folds=5)
    
    print("\n" + "="*60)
    print("MODELING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)

    # Save the model
    save_model(rf_model, "model.pkl")
    
    return rf_model

if __name__ == "__main__":
    model = main()
