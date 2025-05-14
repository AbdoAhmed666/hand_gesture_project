# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# import joblib
# from src.utils.config import MODEL_PATH
# from src.utils.logger import get_logger

# logger = get_logger(__name__)

# def train_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     clf = RandomForestClassifier(n_estimators=100)
#     clf.fit(X_train, y_train)
#     joblib.dump(clf, MODEL_PATH)
#     logger.info("Model trained and saved at %s", MODEL_PATH)
#     return clf, X_test, y_test

from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def save_model(model, path):
    joblib.dump(model, path)
