"""
Automated Data Insights & Visualization Dashboard
Enterprise-Grade Streamlit Application for Business Analytics

Author: Data Science Team
Purpose: Interactive dashboard for automated data insights and KPI visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Automated Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(file):
    """
    Load dataset from CSV file with caching for performance.
    
    Args:
        file: File path (str) or file-like object from Streamlit uploader
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        # Handle both file paths and file upload objects
        if isinstance(file, str):
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def preprocess_data(df):
    """
    Preprocess the dataset: handle missing values and encode categorical features.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    df_processed = df.copy()
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # Convert date columns if present
    date_columns = [col for col in df_processed.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_columns:
        try:
            df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
        except:
            pass
    
    return df_processed


def compute_kpis(df, numeric_cols):
    """
    Compute key performance indicators from the dataset.
    
    Args:
        df (pd.DataFrame): Processed dataset
        numeric_cols (list): List of numeric column names
        
    Returns:
        dict: Dictionary containing computed KPIs
    """
    kpis = {}
    
    # Basic metrics
    kpis['total_records'] = len(df)
    kpis['total_columns'] = len(df.columns)
    
    # Numeric aggregations
    if len(numeric_cols) > 0:
        # Use the first numeric column as primary metric
        primary_col = numeric_cols[0]
        kpis['total_sum'] = df[primary_col].sum()
        kpis['mean_value'] = df[primary_col].mean()
        kpis['median_value'] = df[primary_col].median()
        kpis['std_value'] = df[primary_col].std()
        kpis['min_value'] = df[primary_col].min()
        kpis['max_value'] = df[primary_col].max()
    
    return kpis


def create_trend_chart(df, date_col, value_col, title="Trend Analysis"):
    """
    Create a line chart for trend analysis.
    
    Args:
        df (pd.DataFrame): Dataset
        date_col (str): Date column name
        value_col (str): Value column name
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if date_col not in df.columns:
        return None
    
    # Group by date if available
    try:
        df_grouped = df.groupby(df[date_col].dt.to_period('D'))[value_col].sum().reset_index()
        df_grouped[date_col] = df_grouped[date_col].astype(str)
        
        fig = px.line(
            df_grouped,
            x=date_col,
            y=value_col,
            title=title,
            markers=True
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=value_col,
            hovermode='x unified',
            template='plotly_white'
        )
        return fig
    except:
        return None


def create_bar_chart(df, category_col, value_col, title="Category Comparison"):
    """
    Create a bar chart for category comparison.
    
    Args:
        df (pd.DataFrame): Dataset
        category_col (str): Categorical column name
        value_col (str): Value column name
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if category_col not in df.columns or value_col not in df.columns:
        return None
    
    df_grouped = df.groupby(category_col)[value_col].sum().reset_index()
    df_grouped = df_grouped.sort_values(value_col, ascending=False).head(10)
    
    fig = px.bar(
        df_grouped,
        x=category_col,
        y=value_col,
        title=title,
        color=value_col,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        xaxis_title=category_col,
        yaxis_title=value_col,
        template='plotly_white',
        showlegend=False
    )
    fig.update_xaxes(tickangle=45)
    return fig


def create_distribution_plot(df, numeric_col, title="Distribution Analysis"):
    """
    Create a histogram for distribution analysis.
    
    Args:
        df (pd.DataFrame): Dataset
        numeric_col (str): Numeric column name
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if numeric_col not in df.columns:
        return None
    
    fig = px.histogram(
        df,
        x=numeric_col,
        nbins=30,
        title=title,
        marginal="box"
    )
    fig.update_layout(
        xaxis_title=numeric_col,
        yaxis_title="Frequency",
        template='plotly_white'
    )
    return fig


def generate_insights(df, numeric_cols, categorical_cols, kpis):
    """
    Automatically generate textual insights based on the data.
    
    Args:
        df (pd.DataFrame): Processed dataset
        numeric_cols (list): List of numeric columns
        categorical_cols (list): List of categorical columns
        kpis (dict): Dictionary of computed KPIs
        
    Returns:
        list: List of insight strings
    """
    insights = []
    
    # Basic dataset insights
    insights.append(f"📊 **Dataset Overview**: The dataset contains {kpis['total_records']:,} records across {kpis['total_columns']} features.")
    
    # Numeric insights
    if len(numeric_cols) > 0:
        primary_col = numeric_cols[0]
        mean_val = kpis.get('mean_value', 0)
        std_val = kpis.get('std_value', 0)
        
        insights.append(f"📈 **Primary Metric Analysis**: The average {primary_col} is {mean_val:,.2f} with a standard deviation of {std_val:,.2f}.")
        
        # Check for outliers
        if std_val > mean_val * 0.5:
            insights.append("⚠️ **Variability Alert**: High variability detected in the primary metric, suggesting diverse data patterns.")
    
    # Categorical insights
    if len(categorical_cols) > 0:
        top_cat_col = categorical_cols[0]
        if top_cat_col in df.columns:
            top_category = df[top_cat_col].value_counts().index[0]
            top_count = df[top_cat_col].value_counts().iloc[0]
            top_pct = (top_count / len(df)) * 100
            insights.append(f"🏆 **Top Category**: '{top_category}' represents {top_pct:.1f}% of all records ({top_count:,} records).")
    
    # Data quality insights
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct < 1:
        insights.append("✅ **Data Quality**: Excellent data quality with minimal missing values.")
    elif missing_pct < 5:
        insights.append("✅ **Data Quality**: Good data quality with acceptable levels of missing values.")
    else:
        insights.append("⚠️ **Data Quality**: Significant missing values detected. Consider data cleaning.")
    
    # Trend insights (if date column exists)
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        date_col = date_cols[0]
        value_col = numeric_cols[0]
        try:
            df_sorted = df.sort_values(date_col)
            recent_avg = df_sorted.tail(int(len(df) * 0.2))[value_col].mean()
            older_avg = df_sorted.head(int(len(df) * 0.2))[value_col].mean()
            if recent_avg > older_avg * 1.1:
                insights.append(f"📈 **Trend**: Recent data shows a {((recent_avg/older_avg - 1) * 100):.1f}% increase compared to earlier periods.")
            elif recent_avg < older_avg * 0.9:
                insights.append(f"📉 **Trend**: Recent data shows a {((1 - recent_avg/older_avg) * 100):.1f}% decrease compared to earlier periods.")
        except:
            pass
    
    return insights


def main():
    """
    Main function to run the Streamlit dashboard.
    """
    # Header
    st.markdown('<p class="main-header">📊 Automated Data Insights & Visualization Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enterprise-Grade Analytics Platform for Business Intelligence</p>', unsafe_allow_html=True)
    
    # Sidebar - File Upload
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset (CSV format)",
        type=['csv'],
        help="Upload a CSV file for analysis"
    )
    
    # Default dataset option
    use_default = st.sidebar.checkbox("Use sample dataset instructions", value=False)
    
    if use_default:
        st.sidebar.info("""
        **Sample Dataset Options:**
        1. **Customer Churn Dataset**: Download from Kaggle (Telco Customer Churn)
        2. **Sales Dataset**: Download from Kaggle (Superstore Sales)
        3. **HR Analytics**: Download from Kaggle (HR Analytics)
        
        Place the CSV file in the project directory and upload it above.
        """)
    
    # Auto-load sample data if available and no file uploaded
    sample_data_path = "sample_sales_data.csv"
    if uploaded_file is None and os.path.exists(sample_data_path):
        uploaded_file = sample_data_path
        st.sidebar.success("✅ Auto-loaded sample data!")
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading and processing data..."):
            df = load_data(uploaded_file)
            
            if df is not None:
                # Preprocess data
                df_processed = preprocess_data(df)
                
                # Get column types
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
                date_cols = [col for col in df_processed.columns if 'date' in col.lower() or 'time' in col.lower()]
                
                # Sidebar Filters
                st.sidebar.header("🔍 Filters")
                
                # Categorical filters
                filtered_df = df_processed.copy()
                
                if len(categorical_cols) > 0:
                    for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                        unique_vals = ['All'] + sorted(df_processed[col].unique().tolist())
                        selected = st.sidebar.selectbox(
                            f"Filter by {col}",
                            unique_vals
                        )
                        if selected != 'All':
                            filtered_df = filtered_df[filtered_df[col] == selected]
                
                # Numeric range filters
                if len(numeric_cols) > 0:
                    primary_numeric = numeric_cols[0]
                    min_val = float(df_processed[primary_numeric].min())
                    max_val = float(df_processed[primary_numeric].max())
                    range_vals = st.sidebar.slider(
                        f"Range for {primary_numeric}",
                        min_val,
                        max_val,
                        (min_val, max_val)
                    )
                    filtered_df = filtered_df[
                        (filtered_df[primary_numeric] >= range_vals[0]) &
                        (filtered_df[primary_numeric] <= range_vals[1])
                    ]
                
                # Compute KPIs
                kpis = compute_kpis(filtered_df, numeric_cols)
                
                # Display KPIs
                st.header("📈 Key Performance Indicators")
                if len(numeric_cols) > 0:
                    primary_col = numeric_cols[0]
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Records", f"{kpis['total_records']:,}")
                    with col2:
                        st.metric(f"Total {primary_col}", f"{kpis.get('total_sum', 0):,.2f}")
                    with col3:
                        st.metric(f"Average {primary_col}", f"{kpis.get('mean_value', 0):,.2f}")
                    with col4:
                        st.metric(f"Median {primary_col}", f"{kpis.get('median_value', 0):,.2f}")
                else:
                    st.metric("Total Records", f"{kpis['total_records']:,}")
                
                # Visualizations
                st.header("📊 Data Visualizations")
                
                # Create two columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart for categories
                    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                        bar_fig = create_bar_chart(
                            filtered_df,
                            categorical_cols[0],
                            numeric_cols[0],
                            f"{categorical_cols[0]} vs {numeric_cols[0]}"
                        )
                        if bar_fig:
                            st.plotly_chart(bar_fig, use_container_width=True)
                    
                    # Distribution plot
                    if len(numeric_cols) > 0:
                        dist_fig = create_distribution_plot(
                            filtered_df,
                            numeric_cols[0],
                            f"Distribution of {numeric_cols[0]}"
                        )
                        if dist_fig:
                            st.plotly_chart(dist_fig, use_container_width=True)
                
                with col2:
                    # Trend chart if date column exists
                    if len(date_cols) > 0 and len(numeric_cols) > 0:
                        trend_fig = create_trend_chart(
                            filtered_df,
                            date_cols[0],
                            numeric_cols[0],
                            f"Trend: {numeric_cols[0]} over Time"
                        )
                        if trend_fig:
                            st.plotly_chart(trend_fig, use_container_width=True)
                    
                    # Additional bar chart for second category if available
                    if len(categorical_cols) > 1 and len(numeric_cols) > 0:
                        bar_fig2 = create_bar_chart(
                            filtered_df,
                            categorical_cols[1],
                            numeric_cols[0],
                            f"{categorical_cols[1]} vs {numeric_cols[0]}"
                        )
                        if bar_fig2:
                            st.plotly_chart(bar_fig2, use_container_width=True)
                
                # Full-width additional visualizations
                if len(numeric_cols) > 1:
                    st.subheader("Additional Metrics Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if len(numeric_cols) > 1:
                            dist_fig2 = create_distribution_plot(
                                filtered_df,
                                numeric_cols[1],
                                f"Distribution of {numeric_cols[1]}"
                            )
                            if dist_fig2:
                                st.plotly_chart(dist_fig2, use_container_width=True)
                
                # Insights Section
                st.header("💡 Automated Insights")
                insights = generate_insights(filtered_df, numeric_cols, categorical_cols, kpis)
                
                for insight in insights:
                    st.markdown(f"- {insight}")
                
                # Data Preview
                with st.expander("📋 View Processed Data"):
                    st.dataframe(filtered_df.head(100), use_container_width=True)
                    st.info(f"Showing first 100 rows of {len(filtered_df):,} total records")
                
                # Data Summary
                with st.expander("📊 Data Summary Statistics"):
                    st.dataframe(filtered_df.describe(), use_container_width=True)
                
            else:
                st.error("Failed to load the dataset. Please check the file format.")
    else:
        # Welcome screen
        st.info("👈 Please upload a CSV file using the sidebar to begin analysis.")
        
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Upload Your Data**: Use the sidebar to upload a CSV file
        2. **Automatic Processing**: The app will automatically:
           - Load and preprocess your data
           - Handle missing values
           - Compute key metrics
           - Generate visualizations
           - Provide insights
        
        ### 📊 Recommended Datasets
        
        For best results, use datasets with:
        - Numeric columns (for KPI calculations)
        - Categorical columns (for filtering and comparisons)
        - Date columns (for trend analysis)
        
        ### 💡 Features
        
        - **Automatic KPI Computation**: Key metrics calculated automatically
        - **Interactive Filters**: Filter data by categories and numeric ranges
        - **Professional Visualizations**: Clean, enterprise-ready charts
        - **Automated Insights**: AI-powered insights generation
        - **Performance Optimized**: Cached data loading for fast performance
        """)


if __name__ == "__main__":
    main()

