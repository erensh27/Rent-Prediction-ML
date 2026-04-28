from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
import heapq

app = Flask(__name__)

DATASET_PATH = "House_Rent_Dataset.csv"

# Load trained model
model = joblib.load("rent_prediction_model.pkl")
model_features = model.feature_names_in_

# Load dataset once at startup so the form can be driven by real CSV data
df_raw = pd.read_csv(DATASET_PATH)


def get_form_options():
    """Pull dropdown options + sensible numeric ranges directly from the CSV."""
    return {
        "cities": sorted(df_raw["City"].dropna().unique().tolist()),
        "area_types": sorted(df_raw["Area Type"].dropna().unique().tolist()),
        "furnishings": sorted(df_raw["Furnishing Status"].dropna().unique().tolist()),
        "tenants": sorted(df_raw["Tenant Preferred"].dropna().unique().tolist()),
        "contacts": sorted(df_raw["Point of Contact"].dropna().unique().tolist()),
        "bhk_min": int(df_raw["BHK"].min()),
        "bhk_max": int(df_raw["BHK"].max()),
        "size_min": int(df_raw["Size"].min()),
        "size_max": int(df_raw["Size"].max()),
        "bath_min": int(df_raw["Bathroom"].min()),
        "bath_max": int(df_raw["Bathroom"].max()),
    }


def get_dataset_stats():
    """Headline stats shown on the landing page."""
    return {
        "rows": int(len(df_raw)),
        "cities": int(df_raw["City"].nunique()),
        "avg_rent": int(df_raw["Rent"].mean()),
        "median_rent": int(df_raw["Rent"].median()),
        "min_rent": int(df_raw["Rent"].min()),
        "max_rent": int(df_raw["Rent"].max()),
        "city_avg": {
            city: int(rent)
            for city, rent in df_raw.groupby("City")["Rent"].mean().sort_values().items()
        },
    }


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    submitted = {}

    if request.method == "POST":
        bhk = int(request.form["bhk"])
        size = int(request.form["size"])
        bathroom = int(request.form["bathroom"])
        city = request.form["city"]
        area_type = request.form["area_type"]
        furnishing = request.form["furnishing"]
        tenant = request.form["tenant"]
        contact = request.form["contact"]

        submitted = {
            "bhk": bhk, "size": size, "bathroom": bathroom,
            "city": city, "area_type": area_type,
            "furnishing": furnishing, "tenant": tenant, "contact": contact,
        }

        input_data = {
            "BHK": bhk,
            "Size": size,
            "Bathroom": bathroom,
            f"City_{city}": 1,
            f"Area Type_{area_type}": 1,
            f"Furnishing Status_{furnishing}": 1,
            f"Tenant Preferred_{tenant}": 1,
            f"Point of Contact_{contact}": 1,
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        raw_prediction = float(model.predict(input_df)[0])

        if raw_prediction > 60000:
            prediction = "Above ₹60,000"
        else:
            prediction = f"₹ {int(raw_prediction):,}"

    server_data = {
        "options": get_form_options(),
        "stats": get_dataset_stats(),
        "submitted": submitted,
        "prediction": prediction,
    }
    return render_template(
        "index.html",
        server_data_json=json.dumps(server_data),
    )


def generate_trend_graphs():
    os.makedirs("static", exist_ok=True)

    plt.figure()
    plt.scatter(df_raw["Size"], df_raw["Rent"], alpha=0.4)
    plt.xlabel("Size (sq ft)")
    plt.ylabel("Rent")
    plt.title("Rent vs Size")
    plt.tight_layout()
    plt.savefig("static/rent_vs_size.png")
    plt.close()

    plt.figure()
    df_raw.groupby("City")["Rent"].mean().sort_values().plot(kind="bar")
    plt.ylabel("Average Rent")
    plt.title("Average Rent by City")
    plt.tight_layout()
    plt.savefig("static/rent_by_city.png")
    plt.close()

    plt.figure()
    sns.heatmap(df_raw[["BHK", "Size", "Bathroom", "Rent"]].corr(), annot=True)
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig("static/correlation.png")
    plt.close()


@app.route("/graphs")
def graphs():
    return render_template("graphs.html")


def heap_rent_ranking():
    min_heap = []
    max_heap = []

    for _, row in df_raw.iterrows():
        rent = row["Rent"]
        city = row["City"]
        bhk = row["BHK"]
        heapq.heappush(min_heap, (rent, city, bhk))
        heapq.heappush(max_heap, (-rent, city, bhk))

    cheapest = [heapq.heappop(min_heap) for _ in range(5)]
    premium = [
        (-(item := heapq.heappop(max_heap))[0], item[1], item[2])
        for _ in range(5)
    ]
    return cheapest, premium


@app.route("/recommendations")
def recommendations():
    cheapest, premium = heap_rent_ranking()
    return render_template(
        "recommendations.html",
        cheapest=cheapest,
        premium=premium,
    )


if __name__ == "__main__":
    generate_trend_graphs()
    app.run(debug=True)
