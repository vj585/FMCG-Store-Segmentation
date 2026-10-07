import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# --- Data Loading ---
DATA_DIR = "outputs"
REPORTS_DIR = "reports"

try:
    stores_df = pd.read_csv(os.path.join(DATA_DIR, "store_segments.csv"))
    profiles_df = pd.read_csv(os.path.join(DATA_DIR, "cluster_profiles.csv"))
    recs_df = pd.read_csv(os.path.join(DATA_DIR, "segment_recommendations.csv"))
    outliers_df = pd.read_csv(os.path.join(DATA_DIR, "cluster_outliers.csv"))
    metrics_df = pd.read_csv(os.path.join(REPORTS_DIR, "final_segmentation_metrics.csv"))
    candidates_df = pd.read_csv(os.path.join(REPORTS_DIR, "final_candidate_comparison.csv"))
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Ensure you have run `python run_pipeline.py` to generate the outputs first.")
    exit(1)

# Extract top level metrics
num_stores = len(stores_df)
num_clusters = stores_df['cluster'].nunique()
num_outliers = len(outliers_df[outliers_df['outlier_flag'] == 1])

# Safe metric extraction
def safe_metric(df, col):
    return df[col].iloc[0] if col in df.columns else "N/A"

silhouette_val = safe_metric(metrics_df, "Silhouette")
ari_val = safe_metric(metrics_df, "Stability (Mean ARI)")
largest_cluster_pct = safe_metric(metrics_df, "Largest Cluster %")
algorithm = safe_metric(metrics_df, "Selected Algorithm")
db_val = safe_metric(metrics_df, "Davies-Bouldin")
ch_val = safe_metric(metrics_df, "Calinski-Harabasz")

# Merge outlier flag into stores_df for explorer
stores_df = stores_df.merge(outliers_df[['STORE_CODE', 'outlier_flag', 'distance_to_centroid']], on='STORE_CODE', how='left')

# Format profiles for heatmap (z-scores approximately)
feature_cols = [c for c in profiles_df.columns if c not in ['cluster', 'cluster_label', 'Store Count', 'Percentage', 'Segment Label']]

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "FMCG Retail Store Segmentation"

# --- Reusable Components ---
def kpi_card(title, value, subtitle=None):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="text-muted text-uppercase mb-2", style={"fontSize": "12px", "fontWeight": "bold"}),
            html.H3(f"{value}", className="mb-0", style={"fontWeight": "bold", "color": "#2c3e50"}),
            html.Small(subtitle, className="text-muted") if subtitle else None
        ]),
        className="shadow-sm border-0 mb-4 rounded-3"
    )

# --- Page Layouts ---

# 1. Overview Page
def layout_overview():
    # Chart A: Cluster Size Distribution
    fig_size = px.bar(
        profiles_df, 
        x='cluster_label', 
        y='Percentage', 
        text='Store Count',
        title='Cluster Size Distribution (%)',
        color='cluster_label',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_size.update_layout(showlegend=False, template='plotly_white', xaxis_title="", yaxis_title="Percentage of Stores")
    
    # Chart B: Profile Heatmap
    # Normalize profiles to standard deviations across the clusters for the heatmap
    heatmap_data = profiles_df[feature_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=feature_cols,
        y=profiles_df['cluster_label'],
        colorscale='RdBu',
        zmid=0
    ))
    fig_heatmap.update_layout(title="Standardized Feature Profiles by Cluster", template='plotly_white')
    
    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("Total Stores", num_stores), width=2),
            dbc.Col(kpi_card("Algorithm", algorithm, f"k={num_clusters}"), width=2),
            dbc.Col(kpi_card("Largest Segment", f"{largest_cluster_pct:.1f}%"), width=2),
            dbc.Col(kpi_card("Stability (ARI)", f"{float(ari_val):.4f}" if isinstance(ari_val, (int, float)) else ari_val), width=2),
            dbc.Col(kpi_card("Silhouette", f"{float(silhouette_val):.4f}" if isinstance(silhouette_val, (int, float)) else silhouette_val), width=2),
            dbc.Col(kpi_card("Outliers", num_outliers, "Top 5% distant"), width=2),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_size), className="shadow-sm border-0 rounded-3 mb-4"), width=5),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_heatmap), className="shadow-sm border-0 rounded-3 mb-4"), width=7),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Key Findings", className="fw-bold mb-3"),
                html.P("The segmentation effectively partitions the network into three distinct tiers of stores avoiding the extreme imbalance of earlier K-Means evaluations. The final solution is highly stable (ARI > 0.78), meaning the store groupings are structurally solid and consistent regardless of algorithmic initialization. It effectively isolates high-volume drivers from the balanced core and the volatile tail.", className="text-muted")
            ]), className="shadow-sm border-0 rounded-3"), width=12)
        ])
    ])

# 2. Segments Page
def layout_segments():
    segment_cards = []
    for _, row in recs_df.iterrows():
        c_label = row.get('Segment Label', f"Cluster {row['Cluster']}")
        chars = row.get('Observed Characteristics', 'N/A')
        interpretation = row.get('Business Interpretation', 'N/A')
        action = row.get('Recommended Action', 'N/A')
        
        prof = profiles_df[profiles_df['cluster'] == row['Cluster']].iloc[0]
        count = prof['Store Count']
        pct = prof['Percentage']
        
        card = dbc.Card([
            dbc.CardHeader(html.H4(c_label, className="mb-0 fw-bold"), className="bg-light border-0"),
            dbc.CardBody([
                html.H6(f"{count} Stores ({pct}%)", className="text-muted mb-4"),
                html.Strong("Observed Characteristics:"),
                html.P(chars, className="mb-3"),
                html.Strong("Business Interpretation:"),
                html.P(interpretation, className="mb-3"),
                html.Strong("Recommended Strategy:"),
                html.P(action, className="mb-0 text-primary fw-bold")
            ])
        ], className="shadow-sm border-0 rounded-3 mb-4")
        segment_cards.append(card)
        
    return html.Div([
        html.H3("Segment Business Profiles & Recommendations", className="mb-4 fw-bold"),
        dbc.Row([dbc.Col(c, width=12) for c in segment_cards])
    ])

# 3. Store Explorer Page
def layout_explorer():
    return html.Div([
        html.H3("Store Explorer", className="mb-4 fw-bold"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Filter by Cluster:"),
                    dcc.Dropdown(
                        id='cluster-filter',
                        options=[{'label': 'All Clusters', 'value': 'ALL'}] + 
                                [{'label': c, 'value': c} for c in stores_df['cluster_label'].unique()],
                        value='ALL',
                        clearable=False
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Filter by Outlier Status:"),
                    dcc.Dropdown(
                        id='outlier-filter',
                        options=[
                            {'label': 'All Stores', 'value': 'ALL'},
                            {'label': 'Outliers Only', 'value': 1},
                            {'label': 'Standard Stores', 'value': 0}
                        ],
                        value='ALL',
                        clearable=False
                    )
                ], width=4)
            ], className="mb-4"),
            dash_table.DataTable(
                id='store-table',
                columns=[
                    {"name": "Store Code", "id": "STORE_CODE"},
                    {"name": "Cluster Label", "id": "cluster_label"},
                    {"name": "Distance to Centroid", "id": "distance_to_centroid", "type": "numeric", "format": {"specifier": ".2f"}},
                    {"name": "Is Outlier", "id": "outlier_flag"}
                ],
                data=stores_df.to_dict('records'),
                filter_action="native",
                sort_action="native",
                page_size=15,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '12px', 'fontFamily': 'Inter, sans-serif'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'border': 'none'},
                style_data={'border': 'none', 'borderBottom': '1px solid #e9ecef'},
            )
        ]), className="shadow-sm border-0 rounded-3")
    ])

# 4. Outliers Page
def layout_outliers():
    outliers_only = stores_df[stores_df['outlier_flag'] == 1].sort_values('distance_to_centroid', ascending=False)
    
    return html.Div([
        html.H3("Structural Outliers Analysis", className="mb-4 fw-bold"),
        dbc.Alert([
            html.I(className="bi bi-info-circle-fill me-2"),
            f"Exactly {num_outliers} stores (the top 5% most distant from their respective cluster centroids) have been flagged. ",
            "Outliers are stores whose observed feature profiles are unusual relative to their assigned cluster. ",
            html.Strong("They are not automatically poor-performing stores."),
            " They may represent unusually large flagship stores, structural data gaps, or distinct regional anomalies."
        ], color="info", className="shadow-sm border-0 rounded-3 mb-4"),
        
        dbc.Card(dbc.CardBody([
            dash_table.DataTable(
                columns=[
                    {"name": "Store Code", "id": "STORE_CODE"},
                    {"name": "Assigned Cluster", "id": "cluster_label"},
                    {"name": "Distance to Centroid (Anomaly Score)", "id": "distance_to_centroid", "type": "numeric", "format": {"specifier": ".2f"}}
                ],
                data=outliers_only.to_dict('records'),
                sort_action="native",
                page_size=15,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '12px', 'fontFamily': 'Inter, sans-serif'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'border': 'none'},
                style_data={'border': 'none', 'borderBottom': '1px solid #e9ecef'},
            )
        ]), className="shadow-sm border-0 rounded-3")
    ])

# 5. Methodology Page
def layout_methodology():
    # Diagnostics table
    diag_cols = [{"name": i, "id": i} for i in candidates_df.columns]
    
    return html.Div([
        html.H3("Clustering Methodology & Diagnostics", className="mb-4 fw-bold"),
        
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Data Pipeline", className="fw-bold mb-3"),
                html.Ol([
                    html.Li("Raw transaction data (31M+ rows)"),
                    html.Li("Store-level aggregation"),
                    html.Li("Feature engineering & missing-value median imputation"),
                    html.Li("Log1p transformation (scale metrics) & StandardScaler"),
                    html.Li("K-Means / Agglomerative clustering comparison (k=2..10)"),
                    html.Li("Stability validation via Adjusted Rand Index (ARI)"),
                    html.Li("Final K-Means k=3 business segmentation")
                ], className="mb-0 text-muted")
            ]), className="shadow-sm border-0 rounded-3 mb-4"), width=6),
            
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Why K-Means k=3?", className="fw-bold mb-3"),
                html.P("The earlier k=2 solution was rejected because it produced an extremely imbalanced segmentation, with approximately 98% of stores in one cluster, and had poor stability (mean ARI approximately 0.16).", className="text-muted mb-2"),
                html.P("The final k=3 solution was preferred because it provided:", className="text-muted mb-2"),
                html.Ul([
                    html.Li("Substantially better stability (ARI > 0.78)"),
                    html.Li("More reasonable cluster balance (largest is 55%)"),
                    html.Li("Business interpretability"),
                    html.Li("Acceptable clustering mathematical quality")
                ], className="text-muted mb-0")
            ]), className="shadow-sm border-0 rounded-3 mb-4"), width=6)
        ]),
        
        html.H5("Candidate Diagnostics (K-Means)", className="fw-bold mb-3 mt-2"),
        dbc.Card(dbc.CardBody([
            dash_table.DataTable(
                columns=diag_cols,
                data=candidates_df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '12px', 'fontFamily': 'Inter, sans-serif'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'border': 'none'},
                style_data={'border': 'none', 'borderBottom': '1px solid #e9ecef'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{k} = 3'},
                        'backgroundColor': '#e8f4f8',
                        'fontWeight': 'bold'
                    }
                ]
            )
        ]), className="shadow-sm border-0 rounded-3")
    ])

# --- Main App Layout ---
app.layout = html.Div([
    # Header
    html.Div([
        dbc.Container([
            html.H2("FMCG Retail Store Segmentation", className="text-white mb-0 fw-bold", style={"padding": "20px 0"})
        ])
    ], style={"backgroundColor": "#1a252f"}),
    
    # Navigation & Content
    dbc.Container([
        dcc.Tabs(id="tabs", value='overview', children=[
            dcc.Tab(label='Overview', value='overview', className="custom-tab", selected_className="custom-tab-selected"),
            dcc.Tab(label='Segments', value='segments', className="custom-tab", selected_className="custom-tab-selected"),
            dcc.Tab(label='Store Explorer', value='explorer', className="custom-tab", selected_className="custom-tab-selected"),
            dcc.Tab(label='Outliers', value='outliers', className="custom-tab", selected_className="custom-tab-selected"),
            dcc.Tab(label='Methodology', value='methodology', className="custom-tab", selected_className="custom-tab-selected"),
        ], style={"marginTop": "20px", "marginBottom": "30px"}),
        
        html.Div(id='tab-content')
    ], className="mt-3 pb-5")
], style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})


# --- Callbacks ---
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'overview': return layout_overview()
    elif tab == 'segments': return layout_segments()
    elif tab == 'explorer': return layout_explorer()
    elif tab == 'outliers': return layout_outliers()
    elif tab == 'methodology': return layout_methodology()

@app.callback(
    Output('store-table', 'data'),
    [Input('cluster-filter', 'value'), Input('outlier-filter', 'value')]
)
def update_table(cluster_val, outlier_val):
    dff = stores_df.copy()
    if cluster_val != 'ALL':
        dff = dff[dff['cluster_label'] == cluster_val]
    if outlier_val != 'ALL':
        dff = dff[dff['outlier_flag'] == outlier_val]
    return dff.to_dict('records')

# --- Run Server ---
if __name__ == '__main__':
    # Use standard port 8050
    app.run_server(debug=True, host='0.0.0.0', port=8050)
