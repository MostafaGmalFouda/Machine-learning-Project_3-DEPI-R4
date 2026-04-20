import joblib
from pathlib import Path

def load_all():
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS = BASE_DIR / "models"

    pca = joblib.load(MODELS / "pca.pkl")
    gmm = joblib.load(MODELS / "gmm.pkl")
    lambdas = joblib.load(MODELS / "boxcox_lambdas.pkl")
    features = joblib.load(MODELS / "features.pkl")

    return pca, gmm, lambdas, features