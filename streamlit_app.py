import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go


def calculate_pearson_correlation(data_array, col_a='A', col_b='B'):
    """
    Calculate Pearson correlation coefficient between two columns
    """
    try:
        # Extract the two columns
        a_values = [item[col_a]
                    for item in data_array if col_a in item and col_b in item]
        b_values = [item[col_b]
                    for item in data_array if col_a in item and col_b in item]

        if len(a_values) == 0 or len(b_values) == 0:
            return None, None, "No valid data pairs found"

        if len(a_values) < 2:
            return None, None, "Need at least 2 data points for correlation"

        # Calculate correlation using scipy
        correlation, p_value = pearsonr(a_values, b_values)

        return correlation, p_value, None

    except KeyError as e:
        return None, None, f"Column {e} not found in data"
    except Exception as e:
        return None, None, f"Error calculating correlation: {str(e)}"


def analyze_json_structure(data):
    """
    Analyze the structure of JSON data to help user understand the format
    """
    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]
        if isinstance(first_item, dict):
            columns = list(first_item.keys())
            return True, columns, f"Array of {len(data)} objects"
        else:
            return False, [], "Array doesn't contain objects"
    elif isinstance(data, dict):
        return False, [], "Root is an object, not an array"
    else:
        return False, [], "Invalid JSON structure"


def main():
    st.set_page_config(
        page_title="JSON Correlation Analyzer",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 JSON Correlation Analyzer")
    st.markdown(
        "Upload or paste JSON data to calculate Pearson correlation coefficient between numerical columns")

    # Sidebar for input method selection
    st.sidebar.header("Input Method")
    input_method = st.sidebar.radio("Choose input method:", [
                                    "Paste JSON", "Upload JSON File"])

    data = None

    if input_method == "Paste JSON":
        st.subheader("Paste JSON Data")
        json_input = st.text_area(
            "Enter JSON data:",
            height=200,
            placeholder='[{"A": 1, "B": 2}, {"A": 3, "B": 4}, {"A": 5, "B": 6}]'
        )

        if json_input.strip():
            try:
                data = json.loads(json_input)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {str(e)}")

    else:
        st.subheader("Upload JSON File")
        uploaded_file = st.file_uploader("Choose a JSON file", type="json")

        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON file format: {str(e)}")

    if data is not None:
        # Analyze JSON structure
        is_valid, columns, structure_info = analyze_json_structure(data)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Data Structure Analysis")
            st.info(f"Structure: {structure_info}")

            if is_valid:
                st.success(f"✅ Valid format detected")
                st.write(f"**Available columns:** {', '.join(columns)}")

                # Column selection
                st.subheader("🎯 Select Columns for Correlation")
                numerical_cols = []

                # Try to identify numerical columns
                if len(data) > 0:
                    sample_item = data[0]
                    for col in columns:
                        if col in sample_item:
                            val = sample_item[col]
                            if isinstance(val, (int, float)):
                                numerical_cols.append(col)

                if len(numerical_cols) >= 2:
                    col_a = st.selectbox(
                        "Select Column A:", numerical_cols, index=0)
                    col_b = st.selectbox("Select Column B:",
                                         [col for col in numerical_cols if col != col_a],
                                         index=0)
                else:
                    col_a = st.selectbox("Select Column A:", columns)
                    col_b = st.selectbox("Select Column B:",
                                         [col for col in columns if col != col_a])

                # Calculate correlation
                if st.button("📊 Calculate Correlation", type="primary"):
                    correlation, p_value, error = calculate_pearson_correlation(
                        data, col_a, col_b)

                    if error:
                        st.error(f"❌ {error}")
                    else:
                        # Display results
                        st.subheader("🔍 Correlation Results")

                        # Main metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)

                        with metric_col1:
                            st.metric(
                                "Pearson Correlation",
                                f"{correlation:.4f}",
                                delta=None
                            )

                        with metric_col2:
                            st.metric(
                                "P-value",
                                f"{p_value:.6f}",
                                delta=None
                            )

                        with metric_col3:
                            significance = "Significant" if p_value < 0.05 else "Not Significant"
                            st.metric(
                                "Significance (α=0.05)",
                                significance,
                                delta=None
                            )

                        # Interpretation
                        st.subheader("📝 Interpretation")

                        if abs(correlation) >= 0.7:
                            strength = "Strong"
                            color = "🔴" if correlation > 0 else "🔵"
                        elif abs(correlation) >= 0.3:
                            strength = "Moderate"
                            color = "🟡"
                        else:
                            strength = "Weak"
                            color = "⚪"

                        direction = "positive" if correlation > 0 else "negative"

                        st.info(
                            f"{color} **{strength} {direction} correlation** between {col_a} and {col_b}")

                        # Extract data for visualization
                        df_data = []
                        for item in data:
                            if col_a in item and col_b in item:
                                try:
                                    a_val = float(item[col_a])
                                    b_val = float(item[col_b])
                                    df_data.append(
                                        {col_a: a_val, col_b: b_val})
                                except (ValueError, TypeError):
                                    continue

                        if df_data:
                            df = pd.DataFrame(df_data)

                            # Scatter plot
                            st.subheader("📈 Visualization")

                            fig = px.scatter(
                                df,
                                x=col_a,
                                y=col_b,
                                title=f"Scatter Plot: {col_a} vs {col_b}",
                                trendline="ols"
                            )

                            fig.update_layout(
                                xaxis_title=col_a,
                                yaxis_title=col_b,
                                showlegend=True
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Summary statistics
                            st.subheader("📊 Summary Statistics")
                            summary_stats = df.describe()
                            st.dataframe(
                                summary_stats, use_container_width=True)

                            # Raw data preview
                            st.subheader("👀 Data Preview")
                            st.dataframe(df.head(10), use_container_width=True)

                            if len(df) > 10:
                                st.info(
                                    f"Showing first 10 rows of {len(df)} total data points")
            else:
                st.error(f"❌ {structure_info}")
                st.write(
                    "Expected format: Array of objects with numerical properties")
                st.code(
                    '[{"A": 1, "B": 2}, {"A": 3, "B": 4}, {"A": 5, "B": 6}]', language="json")

        with col2:
            st.subheader("ℹ️ About Pearson Correlation")
            st.markdown("""
            **Pearson Correlation Coefficient (r)**
            - Range: -1 to +1
            - **+1**: Perfect positive correlation
            - **0**: No linear correlation  
            - **-1**: Perfect negative correlation
            
            **Interpretation:**
            - **|r| ≥ 0.7**: Strong correlation
            - **0.3 ≤ |r| < 0.7**: Moderate correlation
            - **|r| < 0.3**: Weak correlation
            
            **P-value:**
            - < 0.05: Statistically significant
            - ≥ 0.05: Not statistically significant
            """)


if __name__ == "__main__":
    main()
