import joblib
from xgboost import XGBClassifier
from .data_pipeline import load_data, train_val_split, scale_features
from .config import MODEL_PATH, SCALER_PATH

def train():
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_val_split(X, y)
    X_train_s, X_val_s, scaler = scale_features(X_train, X_val)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
    model.fit(X_train_s, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

if __name__ == "__main__":
    train()
