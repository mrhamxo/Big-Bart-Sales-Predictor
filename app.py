import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the trained Gradient Boosting model
gbr = joblib.load("models/gbr.pkl")  # Ensure your trained model is saved in this file

st.title("Big Mart Sales Prediction Application")

# Mappings for categorical features
fat_content_map = {0: 'Low Fat', 1: 'Regular'}
item_type_map = {0: 'Dairy', 1: 'Soft Drinks', 2: 'Meat', 3: 'Fruits and Vegetables', 
                 4: 'Household', 5: 'Baking Goods', 6: 'Snack Foods', 7: 'Frozen Foods',
                 8: 'Breakfast', 9: 'Health and Hygiene', 10: 'Hard Drinks', 11: 'Canned',
                 12: 'Breads', 13: 'Starchy Foods', 14: 'Others', 15: 'Seafood'}
outlet_size_map = {0: 'Medium', 1: 'High', 2: 'Small'}
location_type_map = {0: 'Tier 1', 1: 'Tier 3', 2: 'Tier 2'}
outlet_type_map = {0: 'Supermarket Type1', 1: 'Supermarket Type2', 2: 'Grocery Store', 3: 'Supermarket Type3'}

# Sidebar Input Section
st.sidebar.title("Big Mart Sales Predictor")
st.sidebar.write("Input the details below:")

# User Inputs
item_weight = st.sidebar.slider("Item Weight", 0.0, 20.0, 0.2)
item_visibility = st.sidebar.slider("Item Visibility", 0.0, 0.5, 0.1)
item_mrp = st.sidebar.slider("Item MRP", 0.0, 500.0, 10.0)
outlet_age = st.sidebar.slider("Outlet Age", 0, 50, 2)

item_fat_content = st.sidebar.selectbox("Item Fat Content", list(fat_content_map.values()))
item_type = st.sidebar.selectbox("Item Type", list(item_type_map.values()))
outlet_size = st.sidebar.selectbox("Outlet Size", list(outlet_size_map.values()))
outlet_location_type = st.sidebar.selectbox("Outlet Location Type", list(location_type_map.values()))
outlet_type = st.sidebar.selectbox("Outlet Type", list(outlet_type_map.values()))

# Map inputs to encoded values
item_fat_content_encoded = list(fat_content_map.keys())[list(fat_content_map.values()).index(item_fat_content)]
item_type_encoded = list(item_type_map.keys())[list(item_type_map.values()).index(item_type)]
outlet_size_encoded = list(outlet_size_map.keys())[list(outlet_size_map.values()).index(outlet_size)]
outlet_location_type_encoded = list(location_type_map.keys())[list(location_type_map.values()).index(outlet_location_type)]
outlet_type_encoded = list(outlet_type_map.keys())[list(outlet_type_map.values()).index(outlet_type)]

# Prediction Button
if st.sidebar.button("Predict Sales"):
    # Prepare input data with correct column names
    input_data = pd.DataFrame([[
        item_weight, item_fat_content_encoded, item_visibility, item_type_encoded, 
        item_mrp, outlet_size_encoded, outlet_location_type_encoded, outlet_type_encoded, 
        outlet_age
    ]], columns=[
        "Item_Weight", "Item_Fat_Content", "Item_Visibility", "Item_Type", 
        "Item_MRP", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type", 
        "Outlet_Age"
    ])

    # Predict log-transformed sales
    log_predicted_sales = gbr.predict(input_data)
    
    # Reverse log transformation
    predicted_sales = np.expm1(log_predicted_sales)[0]

    # Display Results
    st.write("### Predicted Item Sales")
    st.success(f"{predicted_sales:.2f} /-")

    # Show prediction in context of input
    st.write("#### Input Summary")
    input_summary = pd.DataFrame({
        "Feature": ["Item Weight", "Item Fat Content", "Item Visibility", "Item Type", "Item MRP", 
                    "Outlet Size", "Outlet Location Type", "Outlet Type", "Outlet Age"],
        "Value": [item_weight, item_fat_content, item_visibility, item_type, item_mrp, 
                  outlet_size, outlet_location_type, outlet_type, outlet_age]
    })
    input_summary["Value"] = input_summary["Value"].astype(str)  # Convert all values to strings
    st.table(input_summary)

    # Interactive Chart
    st.write("### Feature Importance")
    feature_names = ["Item Weight", "Item Fat Content", "Item Visibility", "Item Type", "Item MRP", 
                     "Outlet Size", "Outlet Location Type", "Outlet Type", "Outlet Age"]
    feature_importances = gbr.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_names, hue=feature_names, palette="viridis")
    plt.title("Feature Importance for Gradient Boosting Model")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    st.pyplot(plt)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed with ❤️ by [Muhammad Hamza](https://www.linkedin.com/in/muhammad-hamza-khattak/)")
