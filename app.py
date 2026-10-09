import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MataPredict", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    columns = json.load(open("model_columns.json"))
    return model, columns


@st.cache_data
def load_reference():
    return json.load(open("app_reference.json"))


model, columns = load_model()
ref = load_reference()

st.title("MataPredict: will this matatu be overcrowded?")
tab_predict, tab_explore = st.tabs(["Predict a ride", "Explore the data"])

with tab_predict:
    left, right = st.columns(2)
    with left:
        origin = st.selectbox("Town of origin", sorted(ref["origins"]))
        car_type = st.selectbox("Vehicle type", sorted(ref["car_types"]))
        payment = st.selectbox("Payment method", ref["payment_methods"])
    with right:
        hour = st.slider("Hour of departure", 0, 23, 7)
        day = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday",
                                           "Friday", "Saturday", "Sunday"])
        month = st.selectbox("Month", ref["months"])

    if st.button("Predict"):
        info = ref["origins"][origin]
        ride = pd.DataFrame([{
            "route": info["route"],
            "travel_from": origin,
            "hour": hour,
            "minute": 0,
            "day_of_week": day,
            "month": month,
            "car_type": car_type,
            "payment_method": payment,
            "is_peak": int((5 <= hour <= 9) or (16 <= hour <= 20)),
            "max_capacity": ref["car_types"][car_type],
            "route_count": info["route_count"],
            "centroid_lat": info["centroid_lat"],
            "centroid_lon": info["centroid_lon"],
            "gis_missing": info["gis_missing"],
        }])
        row = pd.get_dummies(ride).reindex(columns=columns, fill_value=0)
        prob = model.predict_proba(row)[0, 1]
        st.metric("Chance of overcrowding", f"{prob:.0%}")
        if prob >= 0.5:
            st.error("This ride is likely to be overcrowded.")
        else:
            st.success("This ride is unlikely to be overcrowded.")

with tab_explore:
    by_hour = pd.Series(ref["by_hour"])
    by_hour.index = by_hour.index.astype(int)
    st.subheader("Overcrowding rate by hour of day")
    st.bar_chart(by_hour.sort_index())
    st.subheader("Overcrowding rate by corridor")
    st.bar_chart(pd.Series(ref["by_route"]))