import os

# Base paths
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "House_Rent_Dataset.csv")
MODEL_PATH = os.path.join(HERE, "rent_prediction_model.pkl")
METRICS_PATH = os.path.join(HERE, "model_metrics.json")
IMPORTANCES_PATH = os.path.join(HERE, "feature_importances.csv")
STATIC_DIR = os.path.join(HERE, "static")
FORM_OPTIONS_PATH = os.path.join(HERE, "form_options.json")
STATS_PATH = os.path.join(HERE, "dataset_stats.json")
RECS_PATH = os.path.join(HERE, "recommendations.json")

# Model configurations
RANDOM_STATE = 42
TARGET = "Rent"
NUMERIC_COLS = ["BHK", "Size", "Bathroom", "floor_level", "total_floors"]
CATEGORICAL_COLS = [
    "City",
    "Area Type",
    "Furnishing Status",
    "Tenant Preferred",
    "Point of Contact",
]
DROP_COLS = ["Posted On", "Area Locality"]

# Flask configurations
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))
DEBUG = os.environ.get("DEBUG", "False").lower() in {"1", "true", "yes"}
