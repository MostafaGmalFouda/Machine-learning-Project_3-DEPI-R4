import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import os
import sys
import plotly.express as px
import numpy as np

# ================= PATH =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "customers.csv")
sys.path.append(CURRENT_DIR)

# ================= IMPORTS =================
from utils.load_models import load_all
from utils.preprocessing import apply_boxcox

# ================= LOAD MODELS =================
pca, gmm, lambdas, features = load_all()

# ================= APP =================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)

# ================= LAYOUT =================
app.layout = html.Div([

    html.Div([

        html.H3("📊 Segmentation", className="text-center text-info mt-4 mb-4"),

        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)"}),

        html.A("📊 Dashboard", id="link-dashboard",
               className="nav-text-link active", n_clicks=0),

        html.A("🤖 Prediction", id="link-pred",
               className="nav-text-link", n_clicks=0),

    ], className="mini-sidebar-container"),

    html.Div(id="tabs-content", className="main-body-content")

])

# ================= NAVIGATION =================
@app.callback(
    Output("tabs-content", "children"),
    [Input("link-dashboard", "n_clicks"),
     Input("link-pred", "n_clicks")]
)
def render_page(d, p):

    ctx = callback_context
    page = ctx.triggered_id or "link-dashboard"

    df = pd.read_csv(DATA_PATH)
    df.drop(['Region', 'Channel'], axis=1, inplace=True)

    # ================= DASHBOARD =================
    if page == "link-dashboard":

        # ================= KPIs =================
        kpi_total = len(df)
        kpi_fresh = df["Fresh"].mean()
        kpi_milk = df["Milk"].mean()
        kpi_grocery = df["Grocery"].mean()

        # ================= CLUSTERING =================
        df_scaled = apply_boxcox(df, lambdas)
        reduced = pca.transform(df_scaled)
        clusters = gmm.predict(reduced)
        df["Cluster"] = clusters

        # ================= BASIC CHARTS =================
        cluster_counts = df["Cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]

        fig_pie = px.pie(cluster_counts, names="Cluster", values="Count",
                         title="Customer Segments Distribution")

        fig_corr = px.imshow(df.drop("Cluster", axis=1).corr(),
                             text_auto=True,
                             title="Feature Correlation")

        fig_box = px.box(df, y=df.columns[:-1],
                         title="Spending Distribution")

        # ================= PCA =================
        fig_pca_var = px.bar(
            x=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            y=pca.explained_variance_ratio_,
            text=np.round(pca.explained_variance_ratio_, 3),
            title="PCA Explained Variance"
        )

        pca_weights = pd.DataFrame(pca.components_, columns=features)

        fig_pca_heatmap = px.imshow(
            pca_weights,
            text_auto=True,
            aspect="auto",
            title="PCA Feature Contribution"
        )

        scatter_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
        scatter_df["Cluster"] = clusters

        fig_clusters = px.scatter(
            scatter_df,
            x="PC1",
            y="PC2",
            color=scatter_df["Cluster"].astype(str),
            title="Customer Segments (PCA Space)"
        )

        # ================= 🔥 GAUSSIAN CUSTOMER SEGMENTS =================

        # 1) Gaussian Segments Plot
        fig_gmm = px.scatter(
            scatter_df,
            x="PC1",
            y="PC2",
            color=scatter_df["Cluster"].astype(str),
            opacity=0.6,
            title="🧠 Gaussian Customer Segments (GMM View)"
        )

        # 2) Cluster Size
        fig_cluster_size = px.bar(
            cluster_counts,
            x="Cluster",
            y="Count",
            text="Count",
            title="📊 Cluster Size Distribution"
        )

        # 3) Cluster Profiling
        cluster_profile = df.groupby("Cluster").mean().reset_index()

        fig_cluster_profile = px.imshow(
            cluster_profile.set_index("Cluster"),
            text_auto=True,
            aspect="auto",
            title="📌 Gaussian Cluster Profiling (Mean Behavior)"
        )

        # ================= RETURN =================
        return html.Div([

            html.H2("📊 Customer Dashboard", className="text-white mb-4"),

            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("Total Customers"),
                    html.H3(kpi_total)
                ], className="stat-card-glass p-3"), width=3),

                dbc.Col(html.Div([
                    html.H5("Avg Fresh"),
                    html.H3(f"{kpi_fresh:.0f}")
                ], className="stat-card-glass p-3"), width=3),

                dbc.Col(html.Div([
                    html.H5("Avg Milk"),
                    html.H3(f"{kpi_milk:.0f}")
                ], className="stat-card-glass p-3"), width=3),

                dbc.Col(html.Div([
                    html.H5("Avg Grocery"),
                    html.H3(f"{kpi_grocery:.0f}")
                ], className="stat-card-glass p-3"), width=3),
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_pie), width=6),
                dbc.Col(dcc.Graph(figure=fig_corr), width=6),
            ]),

            html.Br(),

            dcc.Graph(figure=fig_box),

            html.Br(),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_pca_var), width=6),
                dbc.Col(dcc.Graph(figure=fig_pca_heatmap), width=6),
            ]),

            html.Br(),

            dcc.Graph(figure=fig_clusters),

            html.Br(),

            # ================= GAUSSIAN SECTION =================
            html.H3("🧠 Gaussian Customer Segments (GMM)", className="text-info"),

            dcc.Graph(figure=fig_gmm),

            html.Br(),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_cluster_size), width=6),
                dbc.Col(dcc.Graph(figure=fig_cluster_profile), width=6),
            ])

        ])

    # ================= PREDICTION =================
    elif page == "link-pred":

        return html.Div([

            html.H3("🤖 Customer Prediction", className="text-white mb-4"),

            dbc.Row([

                dbc.Col([

                    html.Label("Fresh"),
                    dcc.Input(id="fresh", type="number", className="dark-input"),

                    html.Label("Milk"),
                    dcc.Input(id="milk", type="number", className="dark-input"),

                    html.Label("Grocery"),
                    dcc.Input(id="grocery", type="number", className="dark-input"),

                    html.Label("Frozen"),
                    dcc.Input(id="frozen", type="number", className="dark-input"),

                    html.Label("Detergents"),
                    dcc.Input(id="detergents", type="number", className="dark-input"),

                    html.Label("Delicatessen"),
                    dcc.Input(id="deli", type="number", className="dark-input"),

                    html.Br(), html.Br(),

                    html.Button("🚀 Predict",
                                id="predict-btn",
                                className="predict-button-modern")

                ], width=4),

                dbc.Col([html.Div(id="prediction-output")], width=8)

            ])

        ])

# ================= PREDICT =================
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [
        State("fresh", "value"),
        State("milk", "value"),
        State("grocery", "value"),
        State("frozen", "value"),
        State("detergents", "value"),
        State("deli", "value")
    ],
    prevent_initial_call=True
)
def predict(n, fresh, milk, grocery, frozen, detergents, deli):

    values = [fresh, milk, grocery, frozen, detergents, deli]

    if any(v is None for v in values):
        return dbc.Alert("❌ Please fill all fields", color="danger")

    df = pd.DataFrame([values], columns=features)

    df = apply_boxcox(df, lambdas)

    reduced = pca.transform(df)
    cluster = gmm.predict(reduced)[0]

    if cluster == 0:
        label = "🍽️ Restaurant / Cafe"
        color = "#64ffda"
    else:
        label = "🛒 Retail Store"
        color = "#ff5252"

    return html.Div([
        html.H2(f"Cluster {cluster}", style={"color": color}),
        html.H3(label, style={"color": color})
    ])

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)