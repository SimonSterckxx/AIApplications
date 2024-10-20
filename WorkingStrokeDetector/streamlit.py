import streamlit as st
import json
import os

# Read statistics from the JSON file
stats = {
    "total_falls_detected": 0,
    "total_false_alarms": 0,
    "total_missed_falls": 0
}

# Check if the stats.json file exists and load the data if it does
if os.path.exists("stats.json"):
    with open("stats.json", "r") as f:
        stats = json.load(f)

total_falls_detected = stats["total_falls_detected"]
total_false_alarms = stats["total_false_alarms"]
total_missed_falls = stats["total_missed_falls"]

# Display the statistics in Streamlit
st.title("Fall Detection Statistics")
st.metric("Total Falls Detected", total_falls_detected)
st.metric("False Alarms", total_false_alarms)
st.metric("Missed Falls", total_missed_falls)

# Calculate precision, recall, and F1-score
if total_falls_detected + total_false_alarms > 0:
    precision = total_falls_detected / \
        (total_falls_detected + total_false_alarms)
else:
    precision = 0.0

if total_falls_detected + total_missed_falls > 0:
    recall = total_falls_detected / (total_falls_detected + total_missed_falls)
else:
    recall = 0.0

if precision + recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0.0

st.subheader("Precision")
st.write(f"{precision:.2f}")

st.subheader("Recall")
st.write(f"{recall:.2f}")

st.subheader("F1-Score")
st.write(f"{f1_score:.2f}")
