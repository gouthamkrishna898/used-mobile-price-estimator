
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- Page Setup ----------------
st.set_page_config(
    page_title="Used Mobile Price Estimator",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Used & Refurbished Mobile Price Estimator")
st.write("Enter the phone details below and click Predict price.")

# ---------------- Load Trained Model ----------------
MODEL_PATH = "price_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file 'price_model.pkl' was not found. "
        "Run python model.py first."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# ---------------- Mobile Details Form ----------------
col1, col2 = st.columns(2)

brand = col1.selectbox(
    "Brand",
    [
        "Samsung", "Apple", "Xiaomi", "Realme", "OnePlus",
        "Oppo", "Vivo", "Motorola", "Nokia", "Google"
    ]
)

condition = col2.selectbox(
    "Condition", ["Used", "Refurbished"]
)

ram = col1.selectbox(
    "RAM (GB)", [2, 3, 4, 6, 8, 12, 16], index=4
)

storage = col2.selectbox(
    "Storage (GB)", [32, 64, 128, 256, 512], index=2
)

screen = col1.number_input(
    "Screen size (inches)",
    min_value=5.0,
    max_value=7.2,
    value=6.3,
    step=0.1
)

battery = col2.number_input(
    "Battery (mAh)",
    min_value=2500,
    max_value=6000,
    value=4500,
    step=100
)

main_cam = col1.selectbox(
    "Main camera (MP)",
    [8, 12, 13, 16, 24, 32, 48, 50, 64, 108],
    index=6
)

selfie = col2.selectbox(
    "Selfie camera (MP)",
    [5, 8, 12, 16, 20, 32],
    index=3
)

has_5g = col1.checkbox(
    "5G supported", value=True
)

release_year = col2.number_input(
    "Release year",
    min_value=2016,
    max_value=2025,
    value=2022,
    step=1
)

days_used = col1.number_input(
    "Days used",
    min_value=30,
    max_value=2500,
    value=500,
    step=10
)

new_price = col2.number_input(
    "Original new price (₹)",
    min_value=3000,
    max_value=250000,
    value=30000,
    step=500
)

# ---------------- Prediction and Result Graph ----------------
if st.button("Predict price", type="primary"):

    input_row = pd.DataFrame([{
        "brand": brand,
        "os": "iOS" if brand == "Apple" else "Android",
        "screen_size": screen,
        "ram_gb": ram,
        "storage_gb": storage,
        "battery_mah": battery,
        "main_camera_mp": main_cam,
        "selfie_camera_mp": selfie,
        "has_5g": int(has_5g),
        "days_used": days_used,
        "condition": condition,
        "new_price": new_price,
        "phone_age": 2025 - int(release_year)
    }])

    try:
        # Predict resale price
        predicted_price = float(
            np.expm1(model.predict(input_row)[0])
        )

        predicted_price = max(0.0, predicted_price)

        # Display prediction
        st.success(
            f"Estimated resale price: ₹{predicted_price:,.0f}"
        )

        st.caption(
            "This is an estimate from the trained model, "
            "not a guaranteed market price."
        )

        # Draw a single result graph
        st.subheader("📊 Prediction Result")

        fig, ax = plt.subplots(figsize=(7, 4))

        bars = ax.bar(
            ["Predicted resale price"],
            [predicted_price],
            color="lightskyblue",
            width=0.5
        )

        ax.set_title("Mobile Price Prediction Result")
        ax.set_ylabel("Estimated Price (₹)")

        ax.set_ylim(
            0,
            max(predicted_price * 1.2, 1)
        )

        ax.bar_label(
            bars,
            labels=[f"₹{predicted_price:,.0f}"],
            padding=5
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.35
        )

        ax.set_axisbelow(True)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except Exception as error:
        st.error(f"Could not make prediction: {error}")
