import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import json
warnings.filterwarnings('ignore')


def convert_numpy_to_lists(obj):
    """Convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) for item in obj]
    else:
        return obj


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
    return convert_numpy_to_lists(fig.to_dict())


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
    return convert_numpy_to_lists(fig.to_dict())


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
    return convert_numpy_to_lists(fig.to_dict())


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
    return convert_numpy_to_lists(fig.to_dict())


def run_analysis(params, csv_path):
    # Extract params with defaults
    total_budget = params.get("totalBudget", 1000)
    max_investment_pct = params.get("maxInvestmentPct", 20)  # Add this line
    min_roi = params.get("minROI", 10)
    min_profit = params.get("minProfit", 1)
    max_fba_sellers = params.get("maxFBASellers", 20)
    # Get raw weights and normalize them to sum to 1.0
    sales_weight_raw = params.get("salesWeight", 40)
    rank_weight_raw = params.get("rankWeight", 30)
    fba_weight_raw = params.get("fbaWeight", 20)
    amazon_weight_raw = params.get("amazonWeight", 10)
    
    # Calculate total weight for normalization
    total_weight = sales_weight_raw + rank_weight_raw + fba_weight_raw + amazon_weight_raw
    
    # Normalize weights to sum to 1.0
    sales_weight = sales_weight_raw / total_weight
    rank_weight = rank_weight_raw / total_weight
    fba_weight = fba_weight_raw / total_weight
    amazon_weight = amazon_weight_raw / total_weight
    volatility_weight_30 = params.get("volatilityWeight", 60) / 100
    volatility_weight_90 = 1 - volatility_weight_30
    velocity_volatility_balance = params.get("velocityVolatilityBalance", 70) / 100
    volatility_balance = 1 - velocity_volatility_balance

    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"CSV load error: {e}"}

    # Ensure 'Camel Camel Camel URL' column exists even if missing in input
    if 'Camel Camel Camel URL' not in df.columns:
        df['Camel Camel Camel URL'] = ''

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
    
    # Debug: Print sample normalized values
    print(f"DEBUG: Sample normalized values - Sales: {norm_sales.iloc[0]:.3f}, Rank: {norm_rank.iloc[0]:.3f}, FBA: {norm_fba.iloc[0]:.3f}, Amazon: {amazon_penalty.iloc[0]:.3f}")
    print(f"DEBUG: Sample raw values - Sales: {df['Monthly_Sales_Clean'].iloc[0]:.0f}, Rank: {df['Avg_Rank_Clean'].iloc[0]:.0f}, FBA: {df['FBA_Sellers_Clean'].iloc[0]:.0f}, Amazon Competing: {df['Amazon_Competing'].iloc[0]}")

    df['Velocity_Score'] = (
        norm_sales * sales_weight +
        norm_rank * rank_weight +
        norm_fba * fba_weight +
        amazon_penalty * amazon_weight
    )
    
    # Debug: Print weight information
    print(f"DEBUG: Velocity Score Weights - Sales: {sales_weight:.3f}, Rank: {rank_weight:.3f}, FBA: {fba_weight:.3f}, Amazon: {amazon_weight:.3f}")
    print(f"DEBUG: Total Velocity Weight: {sales_weight + rank_weight + fba_weight + amazon_weight:.3f}")
    print(f"DEBUG: Sample Velocity Score Range: {df['Velocity_Score'].min():.3f} to {df['Velocity_Score'].max():.3f}")
    
    # Debug: Show how weights affect a sample product
    sample_idx = 0
    if len(df) > 0:
        sample_sales = norm_sales.iloc[sample_idx]
        sample_rank = norm_rank.iloc[sample_idx]
        sample_fba = norm_fba.iloc[sample_idx]
        sample_amazon = amazon_penalty.iloc[sample_idx]
        sample_score = df['Velocity_Score'].iloc[sample_idx]
        
        print(f"DEBUG: Sample Product Velocity Score Breakdown:")
        print(f"  - Sales component: {sample_sales:.3f} * {sales_weight:.3f} = {sample_sales * sales_weight:.3f}")
        print(f"  - Rank component: {sample_rank:.3f} * {rank_weight:.3f} = {sample_rank * rank_weight:.3f}")
        print(f"  - FBA component: {sample_fba:.3f} * {fba_weight:.3f} = {sample_fba * fba_weight:.3f}")
        print(f"  - Amazon component: {sample_amazon:.3f} * {amazon_weight:.3f} = {sample_amazon * amazon_weight:.3f}")
        print(f"  - Total Velocity Score: {sample_score:.3f}")

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

    # Select products under budget with max investment per product limit
    df_sorted = df.sort_values('Combined_Score', ascending=False)
    selected_rows = []
    running_total = 0
    max_investment_per_product = total_budget * (max_investment_pct / 100)
    
    for _, row in df_sorted.iterrows():
        price = row['Price_Clean']
        
        # Check if this product exceeds max investment per product
        if price > max_investment_per_product:
            continue  # Skip this product if it's too expensive
        
        # Check if adding this product would exceed total budget
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

    # Ensure ASIN is included in the buy list
    if 'ASIN' in final_df.columns:
        final_df['ASIN'] = final_df['ASIN']
    else:
        final_df['ASIN'] = ''

    # Generate CamelCamelCamel chart URLs for each product
    def generate_camel_urls(asin):
        if not isinstance(asin, str) or asin.strip() == '':
            return '{}'
        base = f'https://charts.camelcamelcamel.com/us/{asin}'
        price_types = ['new', 'used', 'amazon']
        time_periods = ['1m', '3m', '6m', '1y', 'all']
        urls = {ptype: {tp: f"{base}/{ptype}.png?force=1&zero=0&w=692&h=364&desired=false&legend=1&ilt=1&tp={tp}&fo=0&lang=en" for tp in time_periods} for ptype in price_types}
        return json.dumps(urls)
    final_df['Camel_Chart_URLs'] = final_df['ASIN'].apply(generate_camel_urls)

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

    # Create buy list CSV
    buy_list_cols = [
        'Product_Title', 'Price_Clean', 'Profit_Clean', 'ROI_Clean',
        'Velocity_Score', 'Volatility_Score', 'Combined_Score',
        'Monthly_Sales_Clean', 'Avg_Rank_Clean', 'FBA_Sellers_Clean', 'Amazon_Competing'
    ]
    # Add link columns if present
    link_cols = []
    for possible in ['Source URL', 'Amazon URL', 'K URL', 'Camel Camel Camel URL', 'Amazon Seller Central URL']:
        if possible in final_df.columns:
            link_cols.append(possible)
    buy_list_cols.extend(link_cols)
    
    # Add image columns if present
    image_cols = []
    for possible in ['Product Image', 'Amazon Image']:
        if possible in final_df.columns:
            image_cols.append(possible)
    buy_list_cols.extend(image_cols)
    
    # Remove 'K URL' from the buy list columns if present (before creating buy_list_df)
    if 'K URL' in buy_list_cols:
        buy_list_cols.remove('K URL')
    # Add ASIN and Camel_Chart_URLs to buy_list_cols if not present (after buy_list_cols is defined)
    for col in ['ASIN', 'Camel_Chart_URLs']:
        if col not in buy_list_cols:
            buy_list_cols.append(col)
    buy_list_df = final_df[buy_list_cols].copy()
    buy_list_path = 'uploads/buy_list.csv'
    buy_list_df.to_csv(buy_list_path, index=False)

    # Create charts as JSON dicts for frontend Plotly rendering
    charts = {
        "riskReturnChart": create_risk_return_scatter(final_df),
        "quadrantChart": create_quadrant_analysis(final_df),
        "budgetWaterfall": create_budget_waterfall(final_df, total_budget),
        "roiDistribution": create_roi_distribution(final_df),
    }

    return {"metrics": metrics, "charts": charts, "buyListCsvPath": buy_list_path}
