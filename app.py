from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import pandas as pd
import heapq

app = Flask(__name__)

# Load trained model
model = joblib.load("rent_prediction_model.pkl")
model_features = model.feature_names_in_

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        # Get form values
        bhk = int(request.form["bhk"])
        size = int(request.form["size"])
        bathroom = int(request.form["bathroom"])
        city = request.form["city"]
        area_type = request.form["area_type"]
        furnishing = request.form["furnishing"]
        tenant = request.form["tenant"]
        contact = request.form["contact"]

        # Create input dict
        input_data = {
            "BHK": bhk,
            "Size": size,
            "Bathroom": bathroom,
            f"City_{city}": 1,
            f"Area Type_{area_type}": 1,
            f"Furnishing Status_{furnishing}": 1,
            f"Tenant Preferred_{tenant}": 1,
            f"Point of Contact_{contact}": 1
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        raw_prediction = model.predict(input_df)[0]

        if raw_prediction > 60000:
            prediction = "Above ₹60,000"
        else:
            prediction = f"₹ {int(raw_prediction)}"

    return render_template("index.html", prediction=prediction)
def generate_trend_graphs():
    df = pd.read_csv("House_Rent_Dataset.csv")
    os.makedirs("static", exist_ok=True)

    plt.figure()
    plt.scatter(df['Size'], df['Rent'])
    plt.savefig("static/rent_vs_size.png")
    plt.close()

    df.groupby('City')['Rent'].mean().plot(kind='bar')
    plt.savefig("static/rent_by_city.png")
    plt.close()

    sns.heatmap(df[['BHK','Size','Bathroom','Rent']].corr(), annot=True)
    plt.savefig("static/correlation.png")
    plt.close()
@app.route("/graphs")
def graphs():
    return render_template("graphs.html")
def heap_rent_ranking():
    df = pd.read_csv("House_Rent_Dataset.csv")

    min_heap = []
    max_heap = []

    for _, row in df.iterrows():
        rent = row['Rent']
        city = row['City']
        bhk = row['BHK']

        heapq.heappush(min_heap, (rent, city, bhk))
        heapq.heappush(max_heap, (-rent, city, bhk))

    cheapest = [heapq.heappop(min_heap) for _ in range(5)]
    premium = [(-heapq.heappop(max_heap)[0],
                heapq.heappop(max_heap)[1],
                heapq.heappop(max_heap)[2]) for _ in range(5)]

    return cheapest, premium
@app.route("/recommendations")
def recommendations():
    cheapest, premium = heap_rent_ranking()
    return render_template(
        "recommendations.html",
        cheapest=cheapest,
        premium=premium
    )
if __name__ == "__main__":
    generate_trend_graphs()   # ✅ NOW IT EXISTS
    app.run(debug=True)
if __name__ == "__main__":
    generate_trend_graphs()
    app.run(debug=True)

