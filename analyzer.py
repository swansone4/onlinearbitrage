import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


def clean_price(price_str):
    if pd.isna(price_str):
        return 0.0
    price_str = str(price_str).strip()
    try:
        if '(' in price_str and ')' in price_str:
            cleaned = price_str.replace('(', '').replace(')', '').replace('$', '').replace(',', '')
            return -float(cleaned)
        else:
            return float(price_str.replace('$', '').replace(',', ''))
    except ValueError:
        return 0.0

def clean_percentage(pct_str):
    if pd.isna(pct_str):
        return 0.0
    try:
        cleaned = str(pct_str).replace('%', '').strip()
        return float(cleaned)
    except ValueError:
        return 0.0


def create_risk_return_scatter(df):
    fig = px.scatter(df, 
                     x='Volatility_Score', 
                     y='ROI_Clean',
                     size='Price_Clean',
                     color='Velocity_Score',
                     hover_data=['Product_Title', 'Profit_Clean', 'Monthly_Sales_Clean'],
                     title="Risk-Return Analysis: Volatility vs ROI",
                     labels={'Volatility_Score': 'Volatility Score (Higher = Less Volatile)',
                             'ROI_Clean': 'Expected ROI (%)',
                             'Velocity_Score': 'Velocity Score',
                             'Price_Clean': 'Investment ($)'},
                     color_continuous_scale='Viridis')
    
    fig.update_layout(width=700, height=450, title_x=0.5)
    return fig.to_dict()


def create_quadrant_analysis(df):
    velocity_median = df['Velocity_Score'].median()
    volatility_median = df['Volatility_Score'].median()
    
    def get_quadrant(row):
        if row['Velocity_Score'] >= velocity_median and row['Volatility_Score'] >= volatility_median:
            return "Stars"
        elif row['Velocity_Score'] >= velocity_median and row['Volatility_Score'] < volatility_median:
            return "Question Marks"
        elif row['Velocity_Score'] < velocity_median and row['Volatility_Score'] >= volatility_median:
            return "Cash Cows"
        else:
            return "Dogs"
    
    df_viz = df.copy()
    df_viz['Quadrant'] = df_viz.apply(get_quadrant, axis=1)
    
    fig = px.scatter(df_viz,
                     x='Velocity_Score',
                     y='Volatility_Score',
                     color='Quadrant',
                     size='Profit_Clean',
                     hover_data=['Product_Title', 'ROI_Clean', 'Price_Clean'],
                     title="Portfolio Quadrant Analysis: Velocity vs Volatility",
                     labels={'Velocity_Score': 'Velocity Score (Higher = Better)',
                             'Volatility_Score': 'Volatility Score (Higher = Less Volatile)'})
    
    fig.add_hline(y=volatility_median, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=velocity_median, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(width=700, height=450, title_x=0.5)
    return fig.to_dict()


def create_budget_waterfall(df, total_budget):
    df_sorted = df.sort_values('Combined_Score', ascending=False).copy()
    df_sorted['Cumulative_Spend'] = df_sorted['Price_Clean'].cumsum()
    df_sorted['Product_Index'] = range(1, len(df_sorted) + 1)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted['Product_Index'],
        y=df_sorted['Price_Clean'],
        name='Individual Product Cost',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_sorted['Product_Index'],
        y=df_sorted['Cumulative_Spend'],
        mode='lines+markers',
        name='Cumulative Budget Usage',
        line=dict(color='red', width=3),
        yaxis='y2'
    ))
    
    fig.add_hline(y=total_budget, line_dash="dash", line_color="green", 
                  annotation_text=f"Budget Limit: ${total_budget:,.2f}")
    
    fig.update_layout(
        title="Budget Utilization Waterfall",
        xaxis_title="Product Rank (by Combined Score)",
        yaxis_title="Individual Product Cost ($)",
        yaxis2=dict(title="Cumulative Spend ($)", overlaying='y', side='right'),
        width=700, height=450, title_x=0.5
    )
    return fig.to_dict()


def create_roi_distribution(df):
    fig = px.histogram(df,
                       x='ROI_Clean',
                       nbins=20,
                       title="ROI Distribution Across Selected Products",
                       labels={'ROI_Clean': 'Expected ROI (%)',
                               'count': 'Number of Products'},
                       color_discrete_sequence=['skyblue'])
    
    mean_roi = df['ROI_Clean'].mean()
    fig.add_vline(x=mean_roi, line_dash="dash", line_color="red",
                  annotation_text=f"Mean ROI: {mean_roi:.1f}%")
    
    fig.update_layout(width=700, height=450, title_x=0.5)
    return fig.to_dict()


def run_analysis(params, csv_path):
    # Extract params with defaults
    total_budget = params.get("totalBudget", 1000)
    min_roi = params.get("minROI", 10)
    min_profit = params.get("minProfit", 1)
    max_fba_sellers = params.get("maxFBASellers", 20)
    sales_weight = params.get("salesWeight", 40) / 100
    rank_weight = params.get("rankWeight", 30) / 100
    fba_weight = params.get("fbaWeight", 20) / 100
    amazon_weight = params.get("amazonWeight", 10) / 100
    volatility_weight_30 = params.get("volatilityWeight", 60) / 100
    volatility_weight_90 = 1 - volatility_weight_30
    velocity_volatility_balance = params.get("velocityVolatilityBalance", 70) / 100
    volatility_balance = 1 - velocity_volatility_balance

    # Load CSV - adjust path or implement upload later
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"CSV load error: {e}"}
    try:
        df = pd.read_csv(CSV_FILENAME)
    except Exception as e:
        return {"error": f"CSV load error: {e}"}

    # Data cleaning and conversions
    df['Monthly_Sales_Clean'] = pd.to_numeric(df.get('Estimated Monthly Sales', 0), errors='coerce').fillna(0)
    df['Avg_Rank_Clean'] = pd.to_numeric(df.get('Average Rank (90 Days)', np.inf), errors='coerce').fillna(np.inf)
    df['FBA_Sellers_Clean'] = pd.to_numeric(df.get('Competitive FBA Sellers', 0), errors='coerce').fillna(0)
    df['Price_Clean'] = df.get('Price', '$0').apply(clean_price)
    df['ROI_Clean'] = df.get('Gross ROI', '0%').apply(clean_percentage)
    df['Profit_Clean'] = df.get('Gross Profit', '$0').apply(clean_price)
    df['Amazon_Competing'] = df.get('Amazon Sells and In Stock', '').astype(str).str.lower().str.contains('true', na=False)

    # Filter based on user params
    df = df[
        (df['Profit_Clean'] >= min_profit) &
        (df['ROI_Clean'] >= min_roi) &
        (df['FBA_Sellers_Clean'] <= max_fba_sellers) &
        (df['Price_Clean'] > 0) &
        (df['Monthly_Sales_Clean'] > 0) &
        (df['Avg_Rank_Clean'] > 0)
    ]

    if df.empty:
        return {"error": "No products after filtering"}

    # Calculate normalized scores (min-max scaling)
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        return (series - min_val) / (max_val - min_val + 1e-6)

    norm_sales = normalize(df['Monthly_Sales_Clean'])
    norm_rank = 1 - normalize(df['Avg_Rank_Clean'])  # lower rank better
    norm_fba = 1 - normalize(df['FBA_Sellers_Clean'])
    amazon_penalty = (~df['Amazon_Competing']).astype(float)  # 1 if NOT competing, else 0

    df['Velocity_Score'] = (
        norm_sales * sales_weight +
        norm_rank * rank_weight +
        norm_fba * fba_weight +
        amazon_penalty * amazon_weight
    )

    # Volatility calculation
    df['Avg_Price_30d'] = df.get('Average Price (30 days)', '$0').apply(clean_price)
    df['Avg_Price_90d'] = df.get('Average Price (90 days)', '$0').apply(clean_price)
    df['Buy_Box_Price'] = df.get('Amazon Buy Box Price', '$0').apply(clean_price)

    vol_30_90 = abs(df['Avg_Price_30d'] - df['Avg_Price_90d']) / (df['Avg_Price_90d'] + 1e-6)
    vol_bb_30 = abs(df['Buy_Box_Price'] - df['Avg_Price_30d']) / (df['Avg_Price_30d'] + 1e-6)
    df['Volatility_Raw'] = volatility_weight_30 * vol_30_90 + volatility_weight_90 * vol_bb_30

    # Normalize volatility so higher score = less volatile
    df['Volatility_Score'] = 1 - normalize(df['Volatility_Raw'])

    # Combined score with balance
    df['Combined_Score'] = df['Velocity_Score'] * velocity_volatility_balance + df['Volatility_Score'] * volatility_balance

    # Select products under budget
    df_sorted = df.sort_values('Combined_Score', ascending=False)
    selected_rows = []
    running_total = 0
    for _, row in df_sorted.iterrows():
        price = row['Price_Clean']
        if running_total + price <= total_budget:
            selected_rows.append(row)
            running_total += price
        else:
            break

    if not selected_rows:
        return {"error": "No products could be selected within budget"}

    final_df = pd.DataFrame(selected_rows)

    # Add product title if missing
    if 'Title' in final_df.columns:
        final_df['Product_Title'] = final_df['Title']
    else:
        final_df['Product_Title'] = 'Unknown Product'

    # Macro metrics
    metrics = {
        "totalProducts": len(final_df),
        "totalInvestment": round(final_df['Price_Clean'].sum(), 2),
        "expectedProfit": round(final_df['Profit_Clean'].sum(), 2),
        "avgROI": round(final_df['ROI_Clean'].mean(), 2),
        "avgVelocityScore": round(final_df['Velocity_Score'].mean(), 3),
        "avgVolatilityScore": round(final_df['Volatility_Score'].mean(), 3),
        "avgCombinedScore": round(final_df['Combined_Score'].mean(), 3),
        "avgMonthlySales": round(final_df['Monthly_Sales_Clean'].mean(), 2),
    }

    # Create charts as JSON dicts for frontend Plotly rendering
    charts = {
        "riskReturnChart": create_risk_return_scatter(final_df),
        "quadrantChart": create_quadrant_analysis(final_df),
        "budgetWaterfall": create_budget_waterfall(final_df, total_budget),
        "roiDistribution": create_roi_distribution(final_df),
    }

    return {"metrics": metrics, "charts": charts}
