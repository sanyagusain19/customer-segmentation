from flask import Flask, render_template, request
import pandas as pd
import joblib
import plotly.express as px
import plotly.io as pio


app = Flask(__name__)

# Load model and scaler
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

features = [
    "Age",
    "Income",
    "total_spending",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth"
]
df = joblib.load("data.pkl")

scaled_data = scaler.transform(df[features])
df["cluster"] = kmeans.predict(scaled_data)

# ---- brand palette, sampled from the dot logo (teal -> mint -> sky -> blue) ----
CLUSTER_COLORS = {
    0: "#C0E1D8",  # mint
    1: "#A9D5FA",  # sky blue
    2: "#CBE8E3",  # pale mint
    3: "#5FA98C",  # deep teal
    4: "#6FA8DE",  # deep blue
    5: "#E5F2FB",  # palest sky
}
BADGE_TEXT_COLOR = {0: "#2E3D3A", 1: "#2E3D3A", 2: "#2E3D3A", 3: "#FFFFFF", 4: "#FFFFFF", 5: "#2E3D3A"}

segment_names = {
    0: "Low-Value Customers",
    1: "Active Online Customers",
    2: "Premium Senior Customers",
    3: "High-Value Loyal Customers",
    4: "Premium Affluent Customers",
    5: "Budget Browsers"
}

recommendations = {
    0: "Offer discount coupons and personalized promotions to encourage more purchases.",
    1: "Recommend products online, send email campaigns, and cross-sell similar items.",
    2: "Provide premium services, loyalty rewards, and personalized customer support.",
    3: "Retain these loyal customers with VIP memberships and exclusive offers.",
    4: "Offer luxury products, early-access launches, and premium memberships.",
    5: "Use attractive discounts and seasonal offers to convert browsing into purchases."
}


def build_dashboard_fig(highlight=None):
    """Builds the segmentation scatter plot. If `highlight` is given
    (a dict with income, spending, cluster), an extra star marker is
    drawn on top for that specific customer."""

    # Remove extreme outliers only for visualization
    df_plot = df[df["Income"] < 120000]

    color_map = {str(k): v for k, v in CLUSTER_COLORS.items()}

    fig = px.scatter(
        df_plot,
        x="Income",
        y="total_spending",
        color=df_plot["cluster"].astype(str),

        hover_data={
            "Age": True,
            "Income": ":,.0f",
            "total_spending": ":,.0f",
            "NumWebPurchases": True,
            "NumStorePurchases": True
        },

        color_discrete_map=color_map,

        title="Customer Segmentation"
    )

    fig.update_traces(
        marker=dict(
            size=9,
            opacity=0.75,
            line=dict(
                width=0.5,
                color="white"
            )
        )
    )

    if highlight is not None:
        fig.add_scatter(
            x=[highlight["income"]],
            y=[highlight["spending"]],
            mode="markers",
            marker=dict(
                size=22,
                symbol="star",
                color=CLUSTER_COLORS.get(highlight["cluster"], "#5FA98C"),
                line=dict(width=2, color="#2E3D3A")
            ),
            name="This customer"
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.6)",
        font=dict(
            family="Poppins",
            size=14
        ),
        title=dict(
            text="Customer Segmentation by Income & Spending",
            x=0.5,
            font=dict(size=24)
        ),
        legend_title="Cluster",
        height=650
    )

    return fig


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form.get("age"))
    income = float(request.form.get("income"))
    spending = float(request.form.get("total_spending"))
    web_purchase = float(request.form.get("web_purchase"))
    store_purchase = float(request.form.get("store_purchase"))
    web_visit = float(request.form.get("web_visit"))

    customer = [[
        age,
        income,
        spending,
        web_purchase,
        store_purchase,
        web_visit
    ]]

    customer_scaled = scaler.transform(customer)

    cluster = int(kmeans.predict(customer_scaled)[0])
    segment = segment_names[cluster]
    recommendation = recommendations[cluster]

    fig = build_dashboard_fig(highlight={
        "income": income,
        "spending": spending,
        "cluster": cluster
    })
    graph = pio.to_html(fig, full_html=False)

    return render_template(
        "result.html",
        cluster=cluster,
        cluster_color=CLUSTER_COLORS[cluster],
        badge_text_color=BADGE_TEXT_COLOR[cluster],
        age=age,
        income=income,
        spending=spending,
        web_purchase=web_purchase,
        store_purchase=store_purchase,
        web_visit=web_visit,
        segment=segment,
        recommendation=recommendation,
        graph=graph
    )


@app.route("/dashboard")
def dashboard():
    fig = build_dashboard_fig()
    graph = pio.to_html(fig, full_html=False)

    return render_template(
        "dashboard.html",
        graph=graph,
        total_customers=len(df),
        avg_income=round(df["Income"].mean(), 0)
    )


if __name__ == "__main__":
    app.run(debug=True)
