import streamlit as st
import h5py
import pandas as pd
import plotly.graph_objects as go
import os
from ethos.tokenize import Vocabulary  # Replace with actual module
import numpy as np
import plotly.express as px
import re
import numpy as np
import plotly.express as px

# --- Constants ---
DATA_DIR = "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets"
VOCAB_FILE = os.path.join(DATA_DIR, "mimic_vocab_t763.pkl")  # Adjust if needed

# Load Vocabulary
vocab = Vocabulary(VOCAB_FILE)
decode = vocab.decode
print(Vocabulary)
# Get HDF5 files
hdf5_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".hdf5")]

# --- Streamlit UI ---
st.title("📊 Patient Timeline Visualization")

# File Selection
selected_file = st.selectbox("📂 Select HDF5 File:", hdf5_files)
file_path = os.path.join(DATA_DIR, selected_file)

# Function to read HDF5 file
def read_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        datasets = {key: f[key][:] for key in f.keys()}
    return datasets

datasets = read_hdf5(file_path)

# 🔹 Display Dataset Keys
st.markdown("<h3 style='font-size:24px; color:#581845;'>📂 Loaded Datasets:</h3>", unsafe_allow_html=True)
st.write(datasets.keys())

# --- Quartile Visualization Toggle ---
show_quantiles = st.checkbox("📊 Show Quantile Distribution of Event Times")
if show_quantiles:
        st.subheader(" Heatmap of Quantiles (Q1 → Q10)")
        # Add description
        st.markdown(
        "<p style='color:white; font-size:16px; font-weight:bold;'>"
        "The quantile distribution represents the severity of lab test results, where Q1 indicates the lowest severity and Q10 the highest. "
        "These quantiles are derived from lab tests and help in assessing the progression and intensity of patient conditions.<br><br>"
        "Example: Blood Pressure Readings (Systolic BP)<br>"
        "🩸 If the dataset contains systolic blood pressure readings:<br>"
        "➡  Q1 (Lowest Quantile): 90-110 mmHg (Normal BP) → Light Red represents less severe results<br>"
        "➡  Q5 (Mid Quantile): 130-140 mmHg (Pre-Hypertension)<br>"
        "➡  Q10 (Highest Quantile): 180+ mmHg (Hypertensive Crisis) → Dark Red indicates more severe results.<br>"
        "Thus, a **higher quantile** corresponds to more severe conditions.</p>"
        "</p>",
        unsafe_allow_html=True
    )
        
        # Create a simple heatmap
        quantile_matrix = np.array([[i] for i in range(1, 11)])  # 10x1 matrix
        fig_heatmap = px.imshow(quantile_matrix, 
                                color_continuous_scale="Reds",
                                labels=dict(color="Quantile Level Q1(lowest) → Q10(highest)"),
                                y=[f"Q{i}" for i in range(1, 11)],
                                x=["Quantile"],
                                aspect="auto")

        st.plotly_chart(fig_heatmap)


# Define time intervals and their corresponding durations in minutes
time_intervals = {
    "_5m-15m": 5,
    "_15m-1h": 15,
    "_1h-2h": 60,
    "_2h-6h": 2 * 60,
    "_6h-12h": 6 * 60,
    "_12h-1d": 12 * 60,
    "_1d-3d": 24 * 60,
    "_3d-1w": 3 * 24 * 60,
    "_1w-2w": 7 * 24 * 60,
    "_2w-1mt": 2 * 7 * 24 * 60,
    "_1mt-3mt": 30 * 24 * 60,
    "_3mt-6mt": 3 * 30 * 24 * 60,
    "_=6mt": 6 * 30 * 24 * 60
}

# Convert to DataFrame
df = pd.DataFrame({
    "Time Interval": list(time_intervals.keys()),
    "Minutes": list(time_intervals.values())
})

# Custom x-axis labels mapping minutes to readable format
custom_labels = {
    5: "5m",
    15: "15m",
    60: "1h",
    2 * 60: "2h",
    6 * 60: "6h",
    12 * 60: "12h",
    24 * 60: "1d",
    3 * 24 * 60: "3d",
    7 * 24 * 60: "1w",
    2 * 7 * 24 * 60: "2w",
    30 * 24 * 60: "1mt",
    3 * 30 * 24 * 60: "3mt",
    6 * 30 * 24 * 60: "6mt"
}

# Assign colors based on time units
def get_color(value):
    if value < 60:  # Minutes
        return "green"
    elif value < 24 * 60:  # Hours
        return "blue"
    elif value < 7 * 24 * 60:  # Days
        return "orange"
    elif value < 30 * 24 * 60:  # Weeks
        return "purple"
    else:  # Months
        return "red"

df["Color"] = df["Minutes"].apply(get_color)

# Checkbox to toggle visualization
show_time_intervals = st.checkbox("⏳ Show Time Interval Visualization")

if show_time_intervals:
    st.subheader("Time Interval vs Duration")

    # Description with Colored Legend
    st.markdown(
        """
        <div style="color: white; font-size: 16px;">
        <b>🔹 Time Interval Categories</b>  
        <br><b>🟢 Green → Minutes</b> (e.g., 5m, 15m)  
        <br><b>🔵 Blue → Hours</b> (e.g., 1h, 2h, 6h, 12h)  
        <br><b>🟠 Orange → Days</b> (e.g., 1d, 3d)  
        <br><b>🟣 Purple → Weeks</b> (e.g., 1w, 2w)  
        <br><b>🔴 Red → Months</b> (e.g., 1mt, 3mt, 6mt)  
        </div>
        """,
        unsafe_allow_html=True
    )

    # Define color mapping correctly
    color_discrete_map = {
        "green": "green",
        "blue": "blue",
        "orange": "orange",
        "purple": "purple",
        "red": "red"
    }

    # Create the scatter plot
    fig = px.scatter(
        df,
        x="Minutes",
        y="Time Interval",
        color="Color",  # Assign colors correctly
        color_discrete_map=color_discrete_map,  # Fixed syntax
        labels={"Minutes": "Duration", "Time Interval": "Time Ranges"},
        title="Time Interval Duration Mapping",
        template="plotly_dark",
    )

    # Adjust layout for better readability
    fig.update_traces(marker=dict(size=10))  # Adjust point size
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(custom_labels.keys()),
            ticktext=list(custom_labels.values()),
            tickangle=-45,  # Rotate labels for clarity
            type="log"  # Log scale for better spacing
        ),
        margin=dict(l=50, r=50, t=50, b=100),  # Adjust margins for better spacing
    )

    # Display in Streamlit
    st.plotly_chart(fig)






# --- Extracting Patient Data ---
if "patient_ids" in datasets and "times" in datasets and "tokens" in datasets and "patient_data_offsets" in datasets:
    
    # List available patient IDs
    patient_ids = datasets["patient_ids"]
    selected_patient = st.selectbox("🆔 Select Patient ID:", patient_ids)

    # Find patient index
    patient_index = list(patient_ids).index(selected_patient)
    

    # Get patient timeline range
    start_idx = datasets["patient_data_offsets"][patient_index]
    end_idx = datasets["patient_data_offsets"][patient_index + 1] if patient_index + 1 < len(datasets["patient_data_offsets"]) else len(datasets["tokens"])

    # Extract data for the patient
    patient_times = datasets["times"][start_idx:end_idx]
    patient_tokens = datasets["tokens"][start_idx:end_idx]

    def process_events_with_qx(times, tokens, decode):
        """Ensures `_QX` tokens appear adjacent to Lab Test events as a single event."""
        merged_events = []
        i = 0

        while i < len(tokens):
            decoded_token = decode([tokens[i]])[0]

            # Check if the next token is `_QX`
            if i < len(tokens) - 1 and decode([tokens[i + 1]])[0].endswith("_QX"):
                qx_token = decode([tokens[i + 1]])[0]  # Get QX token
                merged_events.append((times[i], f"{decoded_token} ({qx_token})"))
                i += 2  # Skip next token since it's merged
            else:
                merged_events.append((times[i], decoded_token))
                i += 1

        return pd.DataFrame(merged_events, columns=["Time", "Event"])

    # Process patient events
    timeline_df = process_events_with_qx(patient_times, patient_tokens, decode)

    # Round Age and Compute Exact Time Difference
    timeline_df["Age (Years)"] = np.round(timeline_df["Time"]).astype(int)  
    timeline_df["Exact Time Diff"] = timeline_df["Time"].diff().fillna(0)  # Exact Difference in Years

    # Convert to Readable Time Difference (Days, Hours, Minutes)
    timeline_df["Time Diff (Days)"] = (timeline_df["Exact Time Diff"] * 365).round().astype(int)
    timeline_df["Time Diff (Hours)"] = (timeline_df["Exact Time Diff"] * 365 * 24).round().astype(int)
    timeline_df["Time Diff (Minutes)"] = (timeline_df["Exact Time Diff"] * 365 * 24 * 60).round().astype(int)

    # # 🔹 Display Table
    # st.markdown("<h3 style='font-size:22px; color:#581845;'>📋 Decoded Timeline Data:</h3>", unsafe_allow_html=True)
    # st.dataframe(timeline_df, height=400, width=900)
    
    # --- Assign Colors to Unique Events ---
    unique_events = timeline_df["Event"].unique()
    colors = [
        "#FF5733", "#C70039", "#900C3F", "#581845", "#1D8348", "#117A65", "#2874A6",
        "#5B2C6F", "#D4AC0D", "#A93226", "#2471A3", "#D68910", "#28B463", "#7D3C98"
    ]
    event_colors = {event: colors[i % len(colors)] for i, event in enumerate(unique_events)}
    
    st.subheader("About Timeline")
   # st.subheader("Medical Code Categories")

    st.markdown("""
    ## Categorized Breakdown of Medical Codes
    This breakdown helps in identifying ICD codes, lab tests, admissions, discharges, transfers, demographics, and more.

    ### **ICD Codes (Diagnosis)**
    - **Prefix:** `ICD_`
    - **Examples:** `ICD_Heart failure`, `ICD_Pneumonia, unspecified organism`
    - **Description:** Represents diseases, symptoms, and medical conditions.

    ### **ICD Procedure Codes (ICD-PCS)**
    - **Prefix:** `ICD_PCS_`
    - **Examples:** `ICD_PCS_Coronary artery bypass graft`, `ICD_PCS_Knee replacement`
    - **Description:** Represents medical and surgical procedures performed on patients.

    ### **Lab Tests**
    - **Prefix:** `LAB_`
    - **Examples:** `LAB_Hemoglobin`, `LAB_White Blood Cell Count`, `LAB_Creatinine`
    - **Description:** Represents laboratory test results.

    ### **Blood Pressure Measurements**
    - **Prefix:** `BLOOD_PRESSURE_`
    - **Examples:** `BLOOD_PRESSURE_Systolic`, `BLOOD_PRESSURE_Diastolic`
    - **Description:** Represents recorded blood pressure values.

    ### **Admissions & Discharges**
    - **Examples:** `ED_ADMISSION_START`, `DISCHARGED_HOME`, `DISCHARGED_DIED`
    - **Description:** Tracks when a patient is admitted or discharged from a facility.

    ### **Insurance Type**
    - **Prefix:** `INSURANCE_`
    - **Examples:** `INSURANCE_MEDICARE`, `INSURANCE_OTHER`
    - **Description:** Represents the type of insurance coverage.

    ### **Marital Status**
    - **Prefix:** `MARITAL_`
    - **Examples:** `MARITAL_SINGLE`, `MARITAL_MARRIED`
    - **Description:** Denotes the patient's marital status.

    ### **Race/Ethnicity**
    - **Prefix:** `RACE_`
    - **Examples:** `RACE_WHITE`, `RACE_HISPANIC`, `RACE_UNKNOWN`
    - **Description:** Represents racial or ethnic classification.

    ### **Gender/Sex**
    - **Prefix:** `SEX_`
    - **Examples:** `SEX_M`, `SEX_F`
    - **Description:** Represents gender.

    ### **ICU Stays**
    - **Examples:** `ICU_STAY_START`, `ICU_STAY_END`
    - **Description:** Indicates admission and discharge from ICU.

    ### **Transfers**
    - **Prefix:** `TRANSFER_`
    - **Examples:** `TRANSFER_SURG`, `TRANSFER_CMED`
    - **Description:** Tracks patient transfers between departments.

    ### **Procedure Codes (Medical Operations)**
    - **Examples:** `PERC CARDIOVASC PROC W DRUG-ELUTING STENT W/O MCC`
    - **Description:** Represents medical procedures performed on patients.

    ### **SOFA Score (Severity Index)**
    - **Prefix:** `SOFA_`
    - **Examples:** `SOFA_SCORE_Respiration`, `SOFA_SCORE_Liver`
    - **Description:** Sequential Organ Failure Assessment score for critical patients.

    ### **Unknown or Miscellaneous Codes**
    - **Examples:** `UNKNOWN_DRG`, `BMI_UNKNOWN`
    - **Description:** Represents missing or unspecified data.

    """)

    # --- Pagination ---
    events_per_page = 15
    total_pages = (len(timeline_df) // events_per_page) + (1 if len(timeline_df) % events_per_page else 0)

    page_number = st.number_input("📜 Page", min_value=1, max_value=total_pages, step=1) - 1
    paginated_df = timeline_df.iloc[page_number * events_per_page : (page_number + 1) * events_per_page]

    # --- Vertical Timeline Visualization ---
    if not paginated_df.empty:
        st.subheader(f"📈 Page {page_number + 1}/{total_pages}")


        fig = go.Figure()

        fig.add_annotation(
            x=0.5,  
            y=1.10,  
            xref="paper",
            yref="paper",
            text="Age vs Event",
            showarrow=False,
            font=dict(size=20, color="Yellow", family="Arial"),
            align="center"
        )

        for i, row in paginated_df.iterrows():
            event_name = row["Event"].strip()
            time_diff_display = f"{row['Time Diff (Minutes)']} minutes" if row['Time Diff (Minutes)'] != 0 else ""
            # Extract Q value if present
            q_value = ""
            match = re.search(r"(_Q\d+)$", event_name)
            if match:
                q_value = match.group(1)
                event_name = event_name.replace(q_value, "").strip()
            
                # Keep full event name without truncation
            event_text = f"{event_name} {q_value}"


            fig.add_trace(go.Scatter(
                x=[0],  
                y=[-i * 1.5],  
                mode="markers+text",
                marker=dict(size=18, color=event_colors[row["Event"]]),  
                text=f"Age: {round(row['Time'])} | {event_text}",
                textposition="middle right",
                textfont=dict(size=16),
                hoverinfo="text",
                hovertext=f"📅 Full Event: {event_name} {q_value}<br>🕒 Age: {round(row['Time'])}<br>⏳ Time Diff: {row['Time Diff (Minutes)']} minutes"
            ))

        # Layout adjustments
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, autorange="reversed"),
            showlegend=False,
            height=600,
            template="plotly_white"
        )

        st.plotly_chart(fig)
    else:
        st.warning("⚠ No events found for the selected patient.")
