# Rent Prediction ML

Predict monthly rent (in INR) for Indian apartments from BHK, size, city, furnishing status, and other listing features — served as a Flask web app with a JSON API.

## ML Model Architecture

The model is a **`GradientBoostingRegressor`** wrapped in a scikit-learn `Pipeline` that handles all preprocessing and prediction in a single artifact.

```
Raw Input → Feature Engineering → ColumnTransformer → GradientBoostingRegressor → Rent (₹)
```

### Why GradientBoosting?

- Handles non-linear relationships between features and rent (e.g., rent grows faster with size in Mumbai than in Kolkata)
- Naturally robust to outliers compared to linear models
- Built-in feature importance for interpretability
- Performs well on tabular data of this size (~4.7K rows, ~10 features) without requiring extensive feature scaling

### Feature Engineering

| Step | Description |
|------|-------------|
| **Floor parsing** | The raw `Floor` column (e.g. `"Ground out of 2"`, `"3 out of 10"`) is split into `floor_number` (integer, with Ground=0, Basement=-1/-2) and `total_floors` |
| **Locality frequency encoding** | `Area Locality` has ~1,000 unique values — too many for one-hot encoding. A `FrequencyEncoder` maps each locality to its count in the training set; rare localities (<5 occurrences) are grouped into a single "other" value |
| **Interaction features** | `bhk_x_size = BHK × Size` and `bhk_x_bath = BHK × Bathroom` are added so the model can learn interactions between room count and space |
| **Target transform** | Rent is log-transformed (`log1p`) during training to handle the heavy right-skew (range ₹1,200–₹3,500,000). Predictions are exponentiated back to INR |
| **Outlier capping** | Rows above the 95th percentile of rent are removed before training to keep the model robust against extreme listings |

### Categorical Encoding

- Numeric features: passed through as-is (`BHK`, `Size`, `Bathroom`, `floor_number`, `total_floors`, `locality_freq`, `bhk_x_size`, `bhk_x_bath`)
- Categorical features (`City`, `Area Type`, `Furnishing Status`, `Tenant Preferred`, `Point of Contact`): one-hot encoded with `handle_unknown="ignore"` so unseen categories in production don't break predictions

### Hyperparameter Tuning

`RandomizedSearchCV` (5-fold cross-validation, 30 iterations) searches over:
- `n_estimators`: 200–800
- `max_depth`: 3–7
- `learning_rate`: 0.01–0.1
- `subsample`: 0.7–1.0
- `min_samples_leaf`: 1–10

The best configuration is selected by lowest negative MAE and saved to the model artifact.

### Explainability

- **Permutation importance** — how much each feature degrades predictions when shuffled (computed on the test set after training)
- **SHAP values** — per-prediction breakdown showing which features drove the estimate up or down
- Both are saved alongside the model metrics for inspection at `/metrics`

### Benchmark Mode (`--compare`)

An optional CatBoost pipeline is available for comparison. CatBoost handles categorical features natively (no one-hot encoding) and uses ordered boosting to reduce overfitting. When `--compare` is passed, both GBR and CatBoost are trained side-by-side; the model with lower test MAE is saved as the primary model, and the runner-up is saved as a secondary artifact.

> **Note:** Benchmark mode trains two full models with hyperparameter search and can be resource-intensive. On low-end machines, use the default training (GBR only) instead.

## Dataset

- **Source:** House Rent Prediction Dataset (India), included as `House_Rent_Dataset.csv`
- **Size:** 4,746 rows × 12 columns
- **Cities:** Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai
- **Target:** `Rent` (monthly, INR; raw range ₹1,200–₹3,500,000)
- **Outlier handling:** rows above the 95th percentile of rent are removed before training
- **Features used:** numeric (`BHK`, `Size`, `Bathroom`) + categorical (`City`, `Area Type`, `Furnishing Status`, `Tenant Preferred`, `Point of Contact`). `Posted On`, `Area Locality`, and `Floor` are engineered/dropped

## Model Performance

After running `python rent_prediction.py --retrain`, exact values are written to `model_metrics.json`. Typical results for the included dataset:

| Metric          | Value (approx.) |
|-----------------|-----------------|
| Test MAE        | ~₹6,500         |
| Test RMSE       | ~₹9,500         |
| Test R²         | ~0.70           |
| CV R² (5-fold)  | ~0.70 ± 0.02    |

Live numbers from your latest training run are shown at `/metrics` while the server is running.

## Project Structure

```
project/
├── app.py                       # Flask server (routes, prediction, chart generation)
├── rent_prediction.py           # Training script with GBR pipeline + CLI
├── House_Rent_Dataset.csv       # Source data
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── Dockerfile                   # Container build
├── templates/                   # Jinja2 HTML templates
│   ├── index.html               # Prediction form + dataset stats
│   ├── graphs.html              # Trend visualizations
│   ├── recommendations.html     # Cheapest/premium listings
│   └── metrics.html             # Full model metrics page
├── .github/workflows/train.yml  # CI: trains on push to main
└── # Auto-generated after training:
    ├── rent_prediction_model.pkl     # Saved sklearn Pipeline + encoder
    ├── model_metrics.json            # CV + test metrics, SHAP, importances
    ├── feature_importances.csv       # Permutation importance scores
    ├── static/*.png                  # Chart images (rent_vs_size, rent_by_city, etc.)
    └── rent_prediction_model_catboost.pkl  # (only with --compare)
```

## Setup & Installation

### Prerequisites

- Python 3.12+
- pip

### Quick Start

```bash
# 1. Clone the repository
git clone <your-fork-url>
cd Rent-Prediction-ML

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (first-time only)
python rent_prediction.py --retrain

# 5. Start the Flask server
python app.py

# 6. Open http://localhost:5000
```

### Configuration

Optional environment variables (copy `.env.example` to `.env` or set directly):

| Variable | Default     | Description                     |
|----------|-------------|---------------------------------|
| `HOST`   | `0.0.0.0`   | Server bind address             |
| `PORT`   | `5000`      | Server port                     |
| `DEBUG`  | `False`     | Enable Flask debug mode         |

```bash
# Example: run on a different port with debug
HOST=127.0.0.1 PORT=8000 DEBUG=True python app.py
```

## CLI Reference

All commands are run via `rent_prediction.py`:

| Command | Description |
|---------|-------------|
| `python rent_prediction.py` | Train only if no existing model is found |
| `python rent_prediction.py --retrain` | Force retrain and overwrite existing artifacts |
| `python rent_prediction.py --compare` | Train GBR and CatBoost side-by-side, pick the best |

The `--retrain` flag regenerates `rent_prediction_model.pkl`, `model_metrics.json`, and `feature_importances.csv`. Restart `app.py` afterwards so the server picks up the new artifacts.

> **Performance note:** `--compare` trains two full models with hyperparameter search and can use significant CPU/memory. On low-end machines, use the default GBR-only training instead.

## Docker

```bash
# Build the image
docker build -t rent-prediction .

# Run the container
docker run -p 8080:8080 rent-prediction
```

The image installs dependencies, trains the model at build time, and serves via `waitress` on port 8080.

## Testing

Run the unit test suite:

```bash
pytest test_rent_prediction.py -v
```

Tests cover: floor parsing, frequency encoding, interaction features, regression metrics, MAPE edge cases, dataset loading, and pipeline construction (both GBR and CatBoost).

## API Usage

### `POST /api/predict`

JSON in, JSON out:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "BHK": 2,
    "Size": 1000,
    "Bathroom": 2,
    "City": "Mumbai",
    "Area Type": "Super Area",
    "Furnishing Status": "Semi-Furnished",
    "Tenant Preferred": "Bachelors/Family",
    "Point of Contact": "Contact Owner"
  }'
```

Response:

```json
{
  "predicted_rent": 35421,
  "currency": "INR",
  "capped": false,
  "cap_threshold": 200000
}
```

### `GET /health`

Health check endpoint:

```json
{
  "status": "ok",
  "model": "loaded",
  "dataset_rows": 4746,
  "model_version": "2026-06-15T12:00:00+00:00",
  "estimator": "GradientBoostingRegressor"
}
```

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `BHK` | integer | yes | Number of bedrooms (1–6) |
| `Size` | integer | yes | Area in sq ft |
| `Bathroom` | integer | yes | Number of bathrooms |
| `City` | string | yes | One of: Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai |
| `Area Type` | string | yes | One of: Built Area, Carpet Area, Super Area |
| `Furnishing Status` | string | yes | One of: Furnished, Semi-Furnished, Unfurnished |
| `Tenant Preferred` | string | yes | One of: Bachelors, Bachelors/Family, Family |
| `Point of Contact` | string | yes | One of: Contact Agent, Contact Builder, Contact Owner |
| `floor_number` | string | no | Defaults to `"Ground"` |
| `total_floors` | integer | no | Defaults to `1` |

## Web Routes

| Route | Description |
|-------|-------------|
| `/` | Interactive prediction form with dataset stats |
| `/graphs` | Trend visualizations (rent vs size, average rent by city, correlation heatmap, feature importances) |
| `/recommendations` | Top 5 cheapest and top 5 premium listings from the dataset |
| `/metrics` | Full model performance breakdown (test metrics, CV scores, training details) |
| `/health` | JSON health check |

## Retraining

Re-run training when:

- The dataset (`House_Rent_Dataset.csv`) changes
- You modify the pipeline, hyperparameters, or feature list in `rent_prediction.py`
- You upgrade scikit-learn (a saved pickle from one minor version may warn or fail to load on another)

```bash
python rent_prediction.py --retrain
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-improvement`
3. Make your changes and add tests where applicable
4. Run tests: `pytest test_rent_prediction.py -v`
5. Run training to confirm metrics still look reasonable: `python rent_prediction.py --retrain`
6. Open a pull request describing what changed and why

## License

MIT — see [LICENSE](LICENSE).
