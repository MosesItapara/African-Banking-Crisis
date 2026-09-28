from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder,  MinMaxScaler, StandardScaler, OneHotEncoder

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}

def fit_scaler(data, columns, kind="standard"):
    """Fit a scaler to the specified columns of the data"""
    scaler_class = SCALERS[kind]
    scaler = scaler_class()
    values = scaler.fit_transform(data[columns])
    return scaler, values

def save_artifact(obj, path: str) -> None:
    """Save an object to disk using joblib"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_artifact(path: str):
    """Load an object from disk using joblib"""
    return joblib.load(path)

def fit_label_encoder(series):
    """Fit a label encoder on a categorical column"""
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(series)
    return encoder, encoded
