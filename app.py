import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
import warnings
import base64
import time

warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="📊 Web Traffic Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"Background image not found: {image_file}")

add_bg_from_local("images2.jpg")  

@st.cache_resource
def load_trained_model():
    try:
        model = load_model('web_traffic_model.keras')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Model files could not be loaded: {e}")
        return None, None


def create_sequences(data, window_size):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
    return np.array(X)

def inverse_transform_predictions(scaled_predictions, scaler, original_columns, target_col_index):
    dummy_array = np.zeros((len(scaled_predictions), len(original_columns)))
    dummy_array[:, target_col_index] = scaled_predictions.flatten()
    return scaler.inverse_transform(dummy_array)[:, target_col_index]


def main():
    st.markdown("""
    <div style="text-align:center; animation: fadeIn 2s;">
        <h1 style="color:#FFD700;">📊 Web Traffic Forecasting Dashboard</h1>
    </div>
    <style>
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    </style>
    """, unsafe_allow_html=True)

    st.write("Forecast web page views with your trained LSTM model. Adjust settings on the sidebar and visualize predictions.")

    
    model, scaler = load_trained_model()
    if model is None or scaler is None:
        return

    
    try:
        df = pd.read_csv("final_website_stats.csv")
        data = df.drop(columns=['timestamp'])
        data = pd.get_dummies(data, columns=['day_of_week'], drop_first=True)
        original_columns = data.columns.tolist()
        target_col = 'page_views'
        target_idx = original_columns.index(target_col)
    except Exception as e:
        st.error(f"Could not load 'final_website_stats.csv': {e}")
        return

   
    st.sidebar.header("⚙️ Forecast Settings")
    window_size = 30
    n_steps = st.sidebar.slider("Number of days to forecast", 1, 30, 7)

    st.sidebar.markdown("---")
    model_option = st.sidebar.selectbox("Select Model", ["LSTM (Default)"])  # can add more models
    st.sidebar.markdown("Adjust ROI or other parameters below if available:")

    roi_enabled = False
    if "ad_spend" in df.columns:
        st.sidebar.subheader("💰 ROI Settings")
        conv_rate = st.sidebar.number_input("Conversion rate (%)", 0.1, 20.0, 2.0, step=0.1)
        revenue_per_conv = st.sidebar.number_input("Revenue per conversion ($)", 1.0, 500.0, 50.0, step=1.0)
        roi_enabled = True

    
    if len(data) >= window_size:
        last_sequence = data.iloc[-window_size:].values
        scaled_last_sequence = scaler.transform(last_sequence)

        if st.sidebar.button("🚀 Generate Forecast"):
            with st.spinner("Generating forecast..."):
                time.sleep(1)  # simulate loading

                current_sequence = scaled_last_sequence.copy()
                predictions = []

                for _ in range(n_steps):
                    X_pred = current_sequence.reshape(1, window_size, len(original_columns))
                    next_pred = model.predict(X_pred, verbose=0)
                    predictions.append(next_pred[0, 0])

                    new_row = np.zeros((1, len(original_columns)))
                    new_row[0, target_idx] = next_pred[0, 0]
                    current_sequence = np.vstack([current_sequence[1:], new_row])

                predictions = np.array(predictions).reshape(-1, 1)
                original_predictions = inverse_transform_predictions(predictions, scaler, original_columns, target_idx)

                last_date = pd.to_datetime(df['timestamp'].iloc[-1])
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_steps, freq='D')

                results_df = pd.DataFrame({
                    'date': future_dates,
                    'predicted_page_views': original_predictions
                })

                
                st.subheader(f"📈 Forecast for the next {n_steps} days")
                st.dataframe(results_df)

                fig, ax = plt.subplots(figsize=(12, 6))
                historical_dates = pd.to_datetime(df['timestamp'].iloc[-60:])
                historical_views = df['page_views'].iloc[-60:]
                ax.plot(historical_dates, historical_views, label='Historical', color='blue')
                ax.plot(future_dates, original_predictions, label='Forecast', color='red', linestyle='--')
                ax.set_title('Web Traffic Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Page Views')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

               
                st.subheader("🌟 Top 3 Predicted Peak Traffic Days")
                top_days = results_df.sort_values(by="predicted_page_views", ascending=False).head(3)
                st.table(top_days)

                
                if roi_enabled:
                    st.subheader("💹 ROI Forecast")
                    avg_ad_spend = df["ad_spend"].iloc[-30:].mean()
                    future_ad_spend = [avg_ad_spend] * n_steps

                    conversions = original_predictions * (conv_rate / 100.0)
                    revenue = conversions * revenue_per_conv
                    roi = revenue / np.array(future_ad_spend)

                    roi_df = pd.DataFrame({
                        "date": future_dates,
                        "predicted_page_views": original_predictions,
                        "predicted_conversions": conversions,
                        "expected_revenue": revenue,
                        "expected_ad_spend": future_ad_spend,
                        "ROI": roi
                    })
                    st.dataframe(roi_df)

                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.plot(future_dates, roi, marker="o", color="green")
                    ax2.set_title("Predicted ROI over Forecast Horizon")
                    ax2.set_xlabel("Date")
                    ax2.set_ylabel("ROI (Revenue / Ad Spend)")
                    ax2.grid(True)
                    st.pyplot(fig2)

                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download forecast as CSV",
                    data=csv,
                    file_name="web_traffic_forecast.csv",
                    mime="text/csv"
                )
    else:
        st.error("Not enough historical data. Need at least 30 days.")


if __name__ == "__main__":
    main()
