# Rent Prediction ML

Predict monthly rent (in INR) for Indian apartments from BHK, size, city, furnishing, and other listing features — served as a Flask web app with a JSON API.

<!-- add screenshot here -->

## Features

- Trained Gradient Boosting pipeline wrapped in a single sklearn `Pipeline` object (preprocessing + model in one artifact).
- Web UI for interactive predictions, with a static-preview fallback that works without a Python server.
- JSON API endpoint (`POST /api/predict`) for programmatic access.
- Built-in pages for trend graphs, top recommendations, and full model metrics.
- Cross-validation, permutation feature importances, and persisted training metrics.
- 12-factor config via environment variables (`HOST`, `PORT`, `DEBUG`).
- `/health` endpoint for liveness checks.

## Tech Stack

| Layer    | Technology                                         |
|----------|----------------------------------------------------|
| Model    | scikit-learn (GradientBoostingRegressor + Pipeline) |
| Backend  | Flask 3, gunicorn (production)                     |
| Frontend | Vanilla HTML/CSS/JS, server-rendered Jinja2        |
| Data     | pandas, numpy, House_Rent_Dataset.csv              |
| Charts   | matplotlib, seaborn (PNGs generated at startup)    |

## Dataset

- **Source:** House Rent Prediction Dataset (India), included as `House_Rent_Dataset.csv`.
- **Size:** 4,746 rows × 12 columns.
- **Cities:** Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai.
- **Target:** `Rent` (monthly, INR; raw range ₹1,200–₹3,500,000).
- **Outlier handling:** rows above the 95th percentile of rent are removed before training to keep the model robust against extreme listings.
- **Features used:** numeric (`BHK`, `Size`, `Bathroom`) + categorical (`City`, `Area Type`, `Furnishing Status`, `Tenant Preferred`, `Point of Contact`). `Posted On`, `Area Locality`, and `Floor` are dropped.

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
├── app.py                    # Flask server (routes, prediction, charts)
├── rent_prediction.py        # Training script (--retrain flag)
├── House_Rent_Dataset.csv    # Source data
├── model_metrics.json        # Auto-generated after training
├── feature_importances.csv   # Auto-generated after training
├── rent_prediction_model.pkl # Saved sklearn Pipeline (auto-generated)
├── requirements.txt
├── .env.example
├── templates/
│   ├── index.html
│   ├── graphs.html
│   ├── recommendations.html
│   └── metrics.html
└── static/                   # Generated PNG charts (gitignored)
```

## Setup & Installation

1. **Clone the repo**
   ```bash
   git clone <your-fork-url>
   cd Rent-Prediction-ML
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS / Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (first-time only, or any time you want fresh metrics)
   ```bash
   python rent_prediction.py --retrain
   ```
   This writes `rent_prediction_model.pkl`, `model_metrics.json`, and `feature_importances.csv`.

5. **Start the server**
   ```bash
   python app.py
   ```

6. Open <http://localhost:5000>.

   Optional configuration via env vars (or copy `.env.example` to `.env` and load it however you prefer):
   ```bash
   HOST=127.0.0.1 PORT=8000 DEBUG=True python app.py
   ```

## API Usage

`POST /api/predict` — JSON in, JSON out.

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

Other routes:

- `GET /health` — returns `{"status": "ok", "model": "loaded", "dataset_rows": N, "model_version": "..."}`.
- `GET /metrics` — full model metrics page.
- `GET /graphs` — trend graphs and feature importances.
- `GET /recommendations` — top 5 cheapest and top 5 premium listings.

## Retraining the Model

Re-run training when:

- The dataset (`House_Rent_Dataset.csv`) changes.
- You modify the pipeline, hyperparameters, or feature list in `rent_prediction.py`.
- You upgrade scikit-learn (a saved pickle from one minor version may warn or fail to load on another).

```bash
python rent_prediction.py --retrain
```

The script overwrites `rent_prediction_model.pkl`, `model_metrics.json`, and `feature_importances.csv`. Restart `app.py` afterwards so the server picks up the new artifacts.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/my-improvement`.
3. Make your changes and add tests where applicable.
4. Run training to confirm metrics still look reasonable: `python rent_prediction.py --retrain`.
5. Open a pull request describing what changed and why.

## License

MIT — see [LICENSE](LICENSE).
