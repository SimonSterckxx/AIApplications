import streamlit as st


def main():
    st.title("Fall Detection Dashboard")
    st.write("This dashboard shows real-time statistics.")

    # Example charts and metrics
    precision = 100
    recall = 66.67
    false_alert_rate = 0.0

    st.metric(label="Precision", value=f"{precision}%")
    st.metric(label="Recall", value=f"{recall}%")
    st.metric(label="False Alert Rate", value=f"{false_alert_rate}%")


if __name__ == "__main__":
    main()
