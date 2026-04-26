import streamlit as st
import pandas as pd
from transformers import pipeline
from datetime import date

FILE = "leaves.csv"
LEAVE_CATEGORIES = [
    "Sick Leave",
    "Casual Leave",
    "Earned Leave",
    "Emergency Leave",
    "Work From Home",
]

@st.cache_resource
def load_model():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

classifier = load_model()

def load_data():
    try:
        return pd.read_csv(FILE)
    except FileNotFoundError:
        return pd.DataFrame(
            columns=[
                "Name",
                "Category",
                "Start Date",
                "End Date",
                "Days",
                "Reason",
                "AI Suggested Category",
                "Status",
            ]
        )

def save_data(df):
    df.to_csv(FILE, index=False)

def suggest_category(reason):
    result = classifier(reason, LEAVE_CATEGORIES)
    return result["labels"][0]

df = load_data()

menu = st.sidebar.selectbox(
    "Menu",
    ["Apply Leave", "My Leaves", "All Requests"]
)

if menu == "Apply Leave":
    st.title("Leave Application Form")

    name = st.text_input("Employee Name")
    category = st.selectbox("Leave Category", LEAVE_CATEGORIES)
    start_date = st.date_input("Start Date", min_value=date.today())
    end_date = st.date_input("End Date", min_value=start_date)
    reason = st.text_area("Reason for Leave")

    if st.button("Submit Leave"):
        if name and category and reason:
            days = (end_date - start_date).days + 1
            ai_category = suggest_category(reason)

            new_row = {
                "Name": name,
                "Category": category,
                "Start Date": start_date,
                "End Date": end_date,
                "Days": days,
                "Reason": reason,
                "AI Suggested Category": ai_category,
                "Status": "Pending",
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_data(df)

            st.success("Leave applied successfully")
            st.info(f"AI Suggested Category: {ai_category}")
        else:
            st.warning("Please fill all fields")

elif menu == "My Leaves":
    st.title("My Leave History")

    employee_name = st.text_input("Enter your name")

    if employee_name:
        my_data = df[df["Name"].str.lower() == employee_name.lower()]

        if my_data.empty:
            st.warning("No leave records found")
        else:
            st.subheader(f"Leaves for {employee_name}")
            st.dataframe(my_data, use_container_width=True)

            approved_count = len(my_data[my_data["Status"] == "Approved"])
            pending_count = len(my_data[my_data["Status"] == "Pending"])
            rejected_count = len(my_data[my_data["Status"] == "Rejected"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Approved", approved_count)
            col2.metric("Pending", pending_count)
            col3.metric("Rejected", rejected_count)

elif menu == "All Requests":
    st.title("All Leave Requests")

    if df.empty:
        st.info("No leave requests available")
    else:
        st.dataframe(df, use_container_width=True)

        st.subheader("Update Request Status")
        request_index = st.number_input(
            "Enter request row number",
            min_value=0,
            max_value=len(df) - 1,
            step=1
        )
        new_status = st.selectbox("Change Status To", ["Pending", "Approved", "Rejected"])

        if st.button("Update Status"):
            df.at[request_index, "Status"] = new_status
            save_data(df)
            st.success("Status updated successfully")
            st.rerun()
