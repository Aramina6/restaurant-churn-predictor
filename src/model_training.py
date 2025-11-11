import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb

def train_and_save_model(X, y, model_dir='models', results_dir='results'):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Save model
    joblib.dump(model, f'{model_dir}/best_model.pkl')

    # Save metrics
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    with open(f'{results_dir}/evaluation_metrics.txt', 'w') as f:
        f.write("=== Test Set Performance ===\n")
        f.write(report + "\n")
        f.write(f"Test ROC-AUC: {auc:.3f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    return model, X_test, y_test, y_pred, y_proba
