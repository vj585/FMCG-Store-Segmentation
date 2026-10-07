import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_DIR = "outputs"
REPORTS_DIR = "reports"
PROCESSED_DIR = os.path.join("data", "processed")

try:
    stores_df = pd.read_csv(os.path.join(DATA_DIR, "store_segments.csv"))
    profiles_df = pd.read_csv(os.path.join(DATA_DIR, "cluster_profiles.csv"))
    recs_df = pd.read_csv(os.path.join(DATA_DIR, "segment_recommendations.csv"))
    outliers_df = pd.read_csv(os.path.join(DATA_DIR, "cluster_outliers.csv"))
    metrics_df = pd.read_csv(os.path.join(REPORTS_DIR, "final_segmentation_metrics.csv"))
    candidates_df = pd.read_csv(os.path.join(REPORTS_DIR, "final_candidate_comparison.csv"))
    hier_df = pd.read_csv(os.path.join(REPORTS_DIR, "hierarchical_metrics.csv"))
    features_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "store_features_base.parquet"))
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

stores_df = stores_df.merge(features_df, on='STORE_CODE', how='left')
stores_df = stores_df.merge(outliers_df[['STORE_CODE', 'outlier_flag', 'distance_to_centroid']], on='STORE_CODE', how='left')
stores_df['outlier_flag'] = stores_df['outlier_flag'].fillna(0)

feature_cols = [c for c in features_df.columns if c != 'STORE_CODE']
prof_feature_cols = [c for c in profiles_df.columns if c not in ['cluster', 'cluster_label', 'Store Count', 'Percentage', 'Segment Label']]

FEATURE_LABELS = {
    'SPEND_sum_sum': 'Total Spend',
    'SPEND_sum_mean': 'Average Weekly Spend',
    'SPEND_sum_std': 'Spend Volatility',
    'TXN_count_sum': 'Transaction Volume',
    'QUANTITY__sum_sum': 'Total Quantity',
    'BASKET_num': 'Average Basket Size',
    'CUST_unique_sum': 'Unique Customers',
    'PROD_unique_sum': 'Product Variety',
    'PROD_nunique_sum': 'Product Variety',
    'WEEK_unique_sum': 'Active Weeks',
    'SPEND_cv': 'Spend Variability (CV)'
}

def get_label(col):
    return FEATURE_LABELS.get(col, col.replace('_', ' ').title())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(stores_df[feature_cols].fillna(stores_df[feature_cols].median()))
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
stores_df['PCA1'] = components[:, 0]
stores_df['PCA2'] = components[:, 1]

num_stores = len(stores_df)
num_clusters = stores_df['cluster'].nunique()
num_outliers = int(stores_df['outlier_flag'].sum())
ari_val = metrics_df["Stability (Mean ARI)"].iloc[0] if "Stability (Mean ARI)" in metrics_df.columns else "N/A"
sil_val = metrics_df["Silhouette"].iloc[0] if "Silhouette" in metrics_df.columns else "N/A"

PALETTE = ["#005C53", "#042940", "#9FC131", "#DBF227", "#D6D58E"]
cluster_labels_sorted = sorted(stores_df['cluster_label'].unique())
COLOR_MAP = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(cluster_labels_sorted)}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "FRESHBASKET | Store Intelligence"

def kpi_card(title, value, subtitle=None):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="kpi-title"),
            html.H3(f"{value}", className="kpi-value"),
            html.Small(subtitle, className="kpi-subtitle") if subtitle else None
        ]),
        className="custom-card mb-4"
    )

def insight_card(title, implication, action):
    return dbc.Card(dbc.CardBody([
        html.H6("KEY FINDING", className="text-muted fw-bold mb-1", style={"fontSize": "11px"}),
        html.P(title, className="mb-3 fw-bold text-dark"),
        html.H6("BUSINESS IMPLICATION", className="text-muted fw-bold mb-1", style={"fontSize": "11px"}),
        html.P(implication, className="mb-3 text-secondary"),
        html.H6("RECOMMENDED ACTION", className="text-muted fw-bold mb-1", style={"fontSize": "11px"}),
        html.P(action, className="mb-0 text-primary fw-bold")
    ]), className="custom-card mb-4")

def layout_overview():
    fig_size = px.bar(
        profiles_df, 
        x='cluster_label', 
        y='Percentage', 
        text='Percentage',
        color='cluster_label',
        color_discrete_map=COLOR_MAP
    )
    fig_size.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_size.update_layout(
        showlegend=False, 
        template='plotly_white', 
        xaxis_title="", 
        yaxis_title="Percentage of Stores",
        margin=dict(t=20, l=10, r=10, b=10),
        yaxis=dict(showgrid=True, gridcolor='#E4E7EC'),
        xaxis=dict(showgrid=False)
    )
    
    fig_pca = px.scatter(
        stores_df, x='PCA1', y='PCA2', color='cluster_label',
        hover_data=['STORE_CODE'] + feature_cols,
        color_discrete_map=COLOR_MAP
    )
    fig_pca.update_layout(
        template='plotly_white',
        legend_title_text='Segment',
        margin=dict(t=20, l=10, r=10, b=10),
        xaxis=dict(showgrid=True, gridcolor='#E4E7EC', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#E4E7EC', zeroline=False)
    )
    fig_pca.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0)))
    
    heatmap_data = profiles_df[prof_feature_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[get_label(c) for c in prof_feature_cols],
        y=profiles_df['cluster_label'],
        colorscale='GnBu',
        zmid=0
    ))
    fig_heatmap.update_layout(
        template='plotly_white',
        margin=dict(t=20, l=10, r=10, b=10)
    )
    
    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("Total Stores", num_stores), lg=2, md=4, sm=6),
            dbc.Col(kpi_card("Segments", num_clusters), lg=2, md=4, sm=6),
            dbc.Col(kpi_card("Outliers", num_outliers), lg=2, md=4, sm=6),
            dbc.Col(kpi_card("Mean ARI", f"{float(ari_val):.4f}" if isinstance(ari_val, (int, float)) else ari_val, "High Stability"), lg=2, md=6, sm=6),
            dbc.Col(kpi_card("Silhouette", f"{float(sil_val):.4f}" if isinstance(sil_val, (int, float)) else sil_val), lg=2, md=6, sm=6),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("CLUSTER DISTRIBUTION", className="text-muted fw-bold mb-3", style={"fontSize": "11px"}),
                dcc.Graph(figure=fig_size, config={'displayModeBar': False})
            ]), className="custom-card mb-4"), lg=4, md=12),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("STORE SEGMENTS PROJECTION (PCA - For Visualization Only)", className="text-muted fw-bold mb-3", style={"fontSize": "11px"}),
                dcc.Graph(figure=fig_pca)
            ]), className="custom-card mb-4"), lg=8, md=12),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("RELATIVE FEATURE PROFILES BY CLUSTER", className="text-muted fw-bold mb-3", style={"fontSize": "11px"}),
                dcc.Graph(figure=fig_heatmap)
            ]), className="custom-card mb-4"), width=12),
        ])
    ])

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
        color = COLOR_MAP.get(c_label, "#005C53")
        
        card = dbc.Card([
            dbc.CardHeader(
                html.H4(c_label, className="mb-0 fw-bold text-white"), 
                style={"backgroundColor": color, "border": "none"}
            ),
            dbc.CardBody([
                html.H6(f"{count} Stores ({pct}%)", className="text-muted mb-4"),
                dbc.Row([
                    dbc.Col(insight_card(chars, interpretation, action), width=12)
                ])
            ])
        ], className="custom-card mb-4")
        segment_cards.append(card)
        
    return html.Div([
        html.H3("Segment Business Profiles", className="page-title"),
        dbc.Row([dbc.Col(c, lg=4, md=12) for c in segment_cards])
    ])

def layout_explorer():
    table_cols = [{"name": "Store Code", "id": "STORE_CODE"}, {"name": "Segment", "id": "cluster_label"}]
    for c in feature_cols:
        table_cols.append({"name": get_label(c), "id": c, "type": "numeric", "format": {"specifier": ",.1f"}})
    table_cols.append({"name": "Outlier", "id": "outlier_flag"})
        
    return html.Div([
        html.H3("Store Explorer", className="page-title"),
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Filter by Segment:", className="fw-bold text-muted", style={"fontSize":"12px"}),
                            dcc.Dropdown(
                                id='cluster-filter',
                                options=[{'label': 'All Segments', 'value': 'ALL'}] + 
                                        [{'label': c, 'value': c} for c in stores_df['cluster_label'].unique()],
                                value='ALL',
                                clearable=False,
                                className="mb-3"
                            )
                        ], lg=4, md=6),
                        dbc.Col([
                            html.Label("Filter by Outlier Status:", className="fw-bold text-muted", style={"fontSize":"12px"}),
                            dcc.Dropdown(
                                id='outlier-filter',
                                options=[
                                    {'label': 'All Stores', 'value': 'ALL'},
                                    {'label': 'Outliers Only', 'value': 1},
                                    {'label': 'Standard Stores', 'value': 0}
                                ],
                                value='ALL',
                                clearable=False,
                                className="mb-3"
                            )
                        ], lg=4, md=6)
                    ]),
                    dash_table.DataTable(
                        id='store-table',
                        columns=table_cols,
                        data=stores_df.to_dict('records'),
                        filter_action="native",
                        sort_action="native",
                        page_size=15,
                        style_table={'overflowX': 'auto', 'border': 'none'},
                        style_cell={'textAlign': 'left', 'padding': '14px', 'fontFamily': 'Inter, sans-serif', 'fontSize': '13px'},
                        style_header={'backgroundColor': '#F7F8FA', 'fontWeight': 'bold', 'border': 'none', 'color': '#667085'},
                        style_data={'border': 'none', 'borderBottom': '1px solid #E4E7EC', 'color': '#17202A'},
                        row_selectable="single",
                    )
                ]), className="custom-card mb-4")
            ], lg=9, md=12),
            
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Store Details", className="fw-bold mb-4"),
                    html.Div(id="store-detail-content", children=html.P("Select a store from the table to view details.", className="text-muted"))
                ]), className="custom-card", style={"position": "sticky", "top": "20px"})
            ], lg=3, md=12)
        ])
    ])

def layout_outliers():
    outliers_only = stores_df[stores_df['outlier_flag'] == 1].sort_values('distance_to_centroid', ascending=False)
    
    return html.Div([
        html.H3("Structural Outliers", className="page-title"),
        dbc.Alert([
            html.I(className="bi bi-info-circle-fill me-2"),
            f"{num_outliers} atypical stores flagged. ",
            "These stores have feature profiles that are unusual relative to their assigned cluster. ",
            html.Strong("An outlier is not automatically a poor-performing store."),
            " They may warrant investigation into unusual operating patterns or data-coverage issues."
        ], color="info", className="custom-card border-0 mb-4", style={"backgroundColor": "#E8F4F8", "color": "#042940"}),
        
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dash_table.DataTable(
                    columns=[
                        {"name": "Store Code", "id": "STORE_CODE"},
                        {"name": "Assigned Segment", "id": "cluster_label"},
                        {"name": "Distance to Centroid", "id": "distance_to_centroid", "type": "numeric", "format": {"specifier": ".2f"}}
                    ],
                    data=outliers_only.to_dict('records'),
                    sort_action="native",
                    page_size=15,
                    style_table={'overflowX': 'auto', 'border': 'none'},
                    style_cell={'textAlign': 'left', 'padding': '14px', 'fontFamily': 'Inter, sans-serif', 'fontSize': '13px'},
                    style_header={'backgroundColor': '#F7F8FA', 'fontWeight': 'bold', 'border': 'none', 'color': '#667085'},
                    style_data={'border': 'none', 'borderBottom': '1px solid #E4E7EC', 'color': '#17202A'},
                )
            ]), className="custom-card"), lg=8, md=12)
        ])
    ])

def layout_methodology():
    diag_cols = [{"name": i, "id": i} for i in candidates_df.columns]
    hier_cols = [{"name": i, "id": i} for i in hier_df.columns]
    
    return html.Div([
        html.H3("Clustering Methodology", className="page-title"),
        
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("DATA PIPELINE", className="text-muted fw-bold mb-3", style={"fontSize": "11px"}),
                html.Div([
                    html.Div("Raw Transactions", className="process-node"),
                    html.Div("↓", className="process-arrow"),
                    html.Div("Store-Level Aggregation", className="process-node"),
                    html.Div("↓", className="process-arrow"),
                    html.Div("Feature Engineering & Imputation", className="process-node"),
                    html.Div("↓", className="process-arrow"),
                    html.Div("Preprocessing & Scaling (Log1p + StandardScaler)", className="process-node"),
                    html.Div("↓", className="process-arrow"),
                    html.Div("Model Comparison (K-Means vs Agglomerative)", className="process-node"),
                    html.Div("↓", className="process-arrow"),
                    html.Div("Stability Analysis (ARI via Random Seeds)", className="process-node"),
                    html.Div("↓", className="process-arrow"),
                    html.Div("Final K-Means (k=3) & Segmentation", className="process-node-final")
                ], className="text-center")
            ]), className="custom-card mb-4"), lg=4, md=12),
            
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Why K-Means k=3?", className="fw-bold mb-3"),
                    html.P("The earlier k=2 solution was rejected because, although it had a higher Silhouette score, it was severely imbalanced (approximately 98% of stores fell into one cluster), stability was poor, and mean ARI was approximately 0.16.", className="text-secondary mb-3"),
                    html.P("The final k=3 solution was explicitly preferred because it provides:", className="text-secondary mb-2"),
                    html.Ul([
                        html.Li("Substantially better stability (Mean ARI = 0.7853)"),
                        html.Li("A more useful cluster structure"),
                        html.Li("Business interpretability"),
                        html.Li("Acceptable clustering mathematical quality")
                    ], className="text-secondary mb-4"),
                    html.H6("LIMITATIONS", className="text-muted fw-bold mb-2", style={"fontSize": "11px"}),
                    html.P("Clustering is descriptive, not causal. Segments represent observed similarity patterns based on available metrics rather than intrinsic permanent classifications.", className="text-secondary mb-0", style={"fontSize": "12px"})
                ]), className="custom-card mb-4"),
                
                html.H6("K-MEANS CANDIDATE DIAGNOSTICS", className="text-muted fw-bold mb-3 mt-4", style={"fontSize": "11px"}),
                dbc.Card(dbc.CardBody([
                    dash_table.DataTable(
                        columns=diag_cols,
                        data=candidates_df.to_dict('records'),
                        style_table={'overflowX': 'auto', 'border': 'none'},
                        style_cell={'textAlign': 'left', 'padding': '14px', 'fontFamily': 'Inter, sans-serif', 'fontSize': '13px'},
                        style_header={'backgroundColor': '#F7F8FA', 'fontWeight': 'bold', 'border': 'none', 'color': '#667085'},
                        style_data={'border': 'none', 'borderBottom': '1px solid #E4E7EC', 'color': '#17202A'},
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{k} = 3'},
                                'backgroundColor': '#F0F9FA',
                                'fontWeight': 'bold',
                                'color': '#005C53'
                            }
                        ]
                    )
                ]), className="custom-card mb-4"),
                
                html.H6("MODEL COMPARISON (AGGLOMERATIVE WARD)", className="text-muted fw-bold mb-3 mt-4", style={"fontSize": "11px"}),
                dbc.Card(dbc.CardBody([
                    dash_table.DataTable(
                        columns=hier_cols,
                        data=hier_df.to_dict('records'),
                        style_table={'overflowX': 'auto', 'border': 'none'},
                        style_cell={'textAlign': 'left', 'padding': '14px', 'fontFamily': 'Inter, sans-serif', 'fontSize': '13px'},
                        style_header={'backgroundColor': '#F7F8FA', 'fontWeight': 'bold', 'border': 'none', 'color': '#667085'},
                        style_data={'border': 'none', 'borderBottom': '1px solid #E4E7EC', 'color': '#17202A'}
                    )
                ]), className="custom-card")
            ], lg=8, md=12)
        ])
    ])

app.layout = html.Div([
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3([html.Strong("FRESHBASKET "), html.Span("Store Intelligence", style={"fontWeight": "300"})], className="text-white mb-1"),
                    html.P("FMCG Retail Store Segmentation", className="mb-0 text-white-50", style={"fontSize": "14px"})
                ], width=8),
                dbc.Col([
                    html.Div([
                        html.Span("K-Means · 3 Segments · 761 Stores", className="badge bg-light text-dark py-2 px-3 rounded-pill")
                    ], className="text-end mt-2")
                ], width=4)
            ], align="center")
        ], fluid=False)
    ], style={"backgroundColor": "#042940", "padding": "25px 0", "borderBottom": "4px solid #005C53"}),
    
    dbc.Container([
        dcc.Tabs(id="tabs", value='overview', children=[
            dcc.Tab(label='Overview', value='overview', className="custom-tab", selected_className="custom-tab-selected"),
            dcc.Tab(label='Segments', value='segments', className="custom-tab", selected_className="custom-tab-selected"),
            dcc.Tab(label='Store Explorer', value='explorer', className="custom-tab", selected_className="custom-tab-selected"),
            dcc.Tab(label='Outliers', value='outliers', className="custom-tab", selected_className="custom-tab-selected"),
            dcc.Tab(label='Methodology', value='methodology', className="custom-tab", selected_className="custom-tab-selected"),
        ], className="mb-4 custom-tabs-container"),
        
        html.Div(id='tab-content')
    ], className="mt-4 pb-5")
], style={"backgroundColor": "#F7F8FA", "minHeight": "100vh", "fontFamily": "Inter, sans-serif"})


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

@app.callback(
    Output('store-detail-content', 'children'),
    [Input('store-table', 'selected_rows'), Input('store-table', 'data')]
)
def display_store_detail(selected_rows, current_data):
    if not selected_rows or not current_data:
        return html.P("Select a store from the table to view details.", className="text-muted")
    
    row = current_data[selected_rows[0]]
    store_code = row['STORE_CODE']
    segment = row['cluster_label']
    
    rec_row = recs_df[recs_df['Segment Label'] == segment]
    interpretation = rec_row['Business Interpretation'].iloc[0] if not rec_row.empty else "N/A"
    action = rec_row['Recommended Action'].iloc[0] if not rec_row.empty else "N/A"
    
    details = [
        html.H3(f"Store {store_code}", className="mb-3 text-dark fw-bold"),
        html.Div(segment, className="badge mb-4 py-2 px-3", style={"backgroundColor": COLOR_MAP.get(segment, "#000"), "fontSize": "13px"}),
    ]
    
    if row.get('outlier_flag') == 1:
        details.append(html.Div("⚠️ Structural Outlier", className="badge bg-warning text-dark mb-4 ms-2 py-2 px-3"))
        
    details.append(html.H6("FEATURE PROFILE", className="text-muted fw-bold mb-2", style={"fontSize": "11px"}))
    for col in feature_cols:
        val = row.get(col, "N/A")
        val_str = f"{val:,.1f}" if isinstance(val, (int, float)) else str(val)
        details.append(html.Div([
            html.Span(get_label(col), className="text-secondary", style={"fontSize": "13px"}),
            html.Span(val_str, className="float-end fw-bold text-dark")
        ], className="mb-2 border-bottom pb-1"))
        
    details.append(html.Hr(className="my-4"))
    details.append(html.H6("SEGMENT IMPLICATION", className="text-muted fw-bold mb-2", style={"fontSize": "11px"}))
    details.append(html.P(interpretation, className="text-secondary", style={"fontSize": "13px"}))
    
    details.append(html.H6("RECOMMENDED ACTION", className="text-muted fw-bold mb-2 mt-3", style={"fontSize": "11px"}))
    details.append(html.P(action, className="text-primary fw-bold", style={"fontSize": "13px"}))
        
    return html.Div(details)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8051)

