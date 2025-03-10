# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.tsa.arima.model import ARIMA
# import xgboost as xgb
# import joblib
# from pmdarima import auto_arima

# # Load trained XGBoost model
# with open("xgboost_model.pkl", "rb") as xgb_file:
#     xgb_model = pickle.load(xgb_file)

# # Streamlit UI
# st.title("AI-Driven Unemployment Forecasting App")
# st.write("This app predicts unemployment rates using ARIMA (Time Series) and XGBoost (Regression) models.")

# # Upload dataset
# uploaded_file = st.file_uploader("C:/Users/New Ameen Computer/processed_unemployee_data.csv", type=["csv"])
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

#     # Display dataset sample
#     st.subheader("Uploaded Data Sample")
#     st.write(df.head())

#     # Ensure the Date column exists and set it as index
#     if "Date" in df.columns:
#         df["Date"] = pd.to_datetime(df["Date"])
#         df.set_index("Date", inplace=True)
#     else:
#         st.error("The dataset must contain a 'Date' column.")
#         st.stop()

#     # Check if 'estimated_unempl_rate' column exists
#     if "estimated_unempl_rate" not in df.columns:
#         st.error("The dataset must contain 'estimated_unempl_rate' column.")
#         st.stop()

#     # Train or load ARIMA model
#     arima_model = auto_arima(df["estimated_unempl_rate"], seasonal=True, stepwise=True)
#     joblib.dump(arima_model, "arima_model.pkl")
#     st.success("New ARIMA model trained and saved!")

#     # Forecast using ARIMA
#     st.subheader("ARIMA Forecast")
#     periods = st.slider("Select number of months to forecast", 1, 24, 12)
#     forecast_arima = arima_model.predict(n_periods=periods)

#     # XGBoost Prediction
#     st.subheader("XGBoost Predictions")

#     # Ensure features match model training data
#     feature_columns = xgb_model.get_booster().feature_names
#     missing_cols = [col for col in feature_columns if col not in df.columns]

#     for col in missing_cols:
#         df[col] = 0  # Add missing columns with default values

#     # Drop the target column before prediction
#     features = df[feature_columns]
#     xgb_pred = xgb_model.predict(features)

#     # Display Results
#     fig, ax = plt.subplots(figsize=(10, 5))
#     df["estimated_unempl_rate"].plot(ax=ax, label="Actual", color="blue")
    
#     forecast_index = pd.date_range(start=df.index[-1], periods=periods + 1, freq="M")[1:]
#     pd.Series(forecast_arima, index=forecast_index).plot(ax=ax, label="ARIMA Forecast", color="red")
    
#     ax.legend()
#     st.pyplot(fig)

#     # Display XGBoost Predictions
#     df["XGBoost_Pred"] = xgb_pred
#     st.subheader("XGBoost Predictions (Last 5 Rows)")
#     st.write(df[["estimated_unempl_rate", "XGBoost_Pred"]].tail())

# st.write("Developed for AI-driven unemployment forecasting.")













# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.tsa.arima.model import ARIMA
# import xgboost as xgb
# import joblib
# from pmdarima import auto_arima

# # Load trained XGBoost model
# with open("xgboost_model.pkl", "rb") as xgb_file:
#     xgb_model = pickle.load(xgb_file)

# # Streamlit UI
# st.title("AI-Driven Unemployment Forecasting App")
# st.write("This app predicts unemployment rates using ARIMA (Time Series) and XGBoost (Regression) models.")

# # Upload dataset
# uploaded_file = st.file_uploader("Upload Processed CSV File", type=["csv"])
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

#     # Display dataset sample
#     st.subheader("Uploaded Data Sample")
#     st.write(df.head())

#     # Ensure the Date column exists and set it as index
#     if "Date" in df.columns:
#         df["Date"] = pd.to_datetime(df["Date"])
#         df.set_index("Date", inplace=True)
#     else:
#         st.error("The dataset must contain a 'Date' column.")
#         st.stop()

    # # Check if 'estimated_unempl_rate' column exists
    # if "estimated_unempl_rate" not in df.columns:
    #     st.error("The dataset must contain 'estimated_unempl_rate' column.")
    #     st.stop()

    # # Train or load ARIMA model
    # arima_model = auto_arima(df["estimated_unempl_rate"], seasonal=True, stepwise=True)
    # joblib.dump(arima_model, "arima_model.pkl")
    # st.success("New ARIMA model trained and saved!")

    # # ARIMA Forecast
    # st.subheader("ARIMA Forecast")
    # periods = st.slider("Select number of months to forecast", 1, 24, 12)
    # forecast_arima = arima_model.predict(n_periods=periods)

    # # Display ARIMA Forecast
    # forecast_dates = pd.date_range(start=df.index[-1], periods=periods + 1, freq="M")[1:]
    # forecast_df = pd.DataFrame({"Date": forecast_dates, "ARIMA_Forecast": forecast_arima})
    # st.write(forecast_df)

#     # XGBoost Predictions
#     st.subheader("XGBoost Predictions")

#     # Identify one-hot encoded region columns
#     region_columns = [col for col in df.columns if col.startswith("Region_")]
#     feature_columns = xgb_model.get_booster().feature_names
#     other_features = [col for col in feature_columns if col not in region_columns + ["Urban", "Rural"]]

#     # Create a dropdown for region selection
#     selected_region = st.selectbox("Select Region", [col.replace("Region_", "") for col in region_columns])

#     # Create a dropdown for area type (Urban/Rural)
#     selected_area = st.selectbox("Select Area Type", ["Urban", "Rural"])

#     # Create user input dictionary
#     user_input = {col: st.number_input(f"Enter value for {col}", value=float(df[col].mean())) for col in other_features}

#     # Convert selected region into one-hot encoding
#     user_region_input = {col: 0 for col in region_columns}
#     selected_region_col = f"Region_{selected_region}"
#     if selected_region_col in user_region_input:
#         user_region_input[selected_region_col] = 1  # Set the selected region to 1

#     # Convert Urban/Rural into one-hot encoding
#     user_input["Urban"] = 1 if selected_area == "Urban" else 0
#     user_input["Rural"] = 1 if selected_area == "Rural" else 0

#     # Combine all user inputs
#     user_input.update(user_region_input)

#     # Predict with XGBoost
#     user_df = pd.DataFrame([user_input])
#     xgb_custom_pred = xgb_model.predict(user_df)
#     st.write(f"Predicted Unemployment Rate: **{xgb_custom_pred[0]:.2f}%**")

# st.write("Developed for AI-driven unemployment forecasting.")














import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from pmdarima import auto_arima
import joblib
# Load trained XGBoost model
with open("xgboost_model.pkl", "rb") as xgb_file:
    xgb_model = pickle.load(xgb_file)

# Get expected feature names from trained model
expected_features = xgb_model.get_booster().feature_names

st.title("Unemployment Rate Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload Processed CSV File", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure the Date column exists and set it as index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
    
    st.write("Dataset Sample:")
    st.write(df.head())




    # Check if 'estimated_unempl_rate' column exists
    if "estimated_unempl_rate" not in df.columns:
        st.error("The dataset must contain 'estimated_unempl_rate' column.")
        st.stop()

    # Train or load ARIMA model
    arima_model = auto_arima(df["estimated_unempl_rate"], seasonal=True, stepwise=True)
    joblib.dump(arima_model, "arima_model.pkl")
    st.success("New ARIMA model trained and saved!")

    # ARIMA Forecast
    st.subheader("ARIMA Forecast")
    periods = st.slider("Select number of months to forecast", 1, 24, 12)
    forecast_arima = arima_model.predict(n_periods=periods)

    # Display ARIMA Forecast
    forecast_dates = pd.date_range(start=df.index[-1], periods=periods + 1, freq="M")[1:]
    forecast_df = pd.DataFrame({"Date": forecast_dates, "ARIMA_Forecast": forecast_arima})
    st.write(forecast_df)






    # Identify user-input features (excluding one-hot encoded regions)
    region_columns = [col for col in df.columns if col.startswith("Region_")]
    other_features = [col for col in expected_features if col not in region_columns]

    # Dropdown for region selection
    selected_region = st.selectbox("Select Region", [col.replace("Region_", "") for col in region_columns])

    # Dropdown for area selection (Urban/Rural)
    selected_area = st.selectbox("Select Area Type", ["Urban", "Rural"])

    # User input for numerical features
    user_input = {}
    for col in other_features:
        if col in df.columns:  # If feature exists in dataset
            user_input[col] = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
        else:  # Handle missing features like 'diff', 'diff2'
            # st.warning(f"Feature '{col}' is missing in the dataset. Using default value 0.")
            user_input[col] = 0  # Default value for missing features

    # Convert selected region into one-hot encoding
    user_region_input = {col: 0 for col in region_columns}
    selected_region_col = f"Region_{selected_region}"
    if selected_region_col in user_region_input:
        user_region_input[selected_region_col] = 1  # Set selected region to 1

    # Convert Urban/Rural into one-hot encoding
    user_input["Area_Rural"] = 1 if selected_area == "Rural" else 0
    user_input["Area_Urban"] = 1 if selected_area == "Urban" else 0

    # Merge all user inputs into a single dictionary
    user_input.update(user_region_input)

    # Ensure input data matches the trained model features
    user_df = pd.DataFrame([user_input])
    user_df = user_df[expected_features]  # Ensure correct column order

    # Make predictions with XGBoost
    xgb_pred = xgb_model.predict(user_df)
    st.write(f"Predicted Unemployment Rate: **{xgb_pred[0]:.2f}%**")
