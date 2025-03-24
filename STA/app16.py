import streamlit as st
import h5py
import pandas as pd
import plotly.graph_objects as go
import os
from ethos.tokenize import Vocabulary  # Replace with actual module
import numpy as np
import plotly.express as px 
import re

# --- Constants ---
DATA_DIR = "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets"
VOCAB_FILE = os.path.join(DATA_DIR, "mimic_vocab_t763.pkl")  # Adjust if needed

# Load Vocabulary
vocab = Vocabulary(VOCAB_FILE)
decode = vocab.decode

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

# 🔹 **Enhanced Font Size for Dataset Names**
st.markdown("<h3 style='font-size:24px; color:##581845;'>📂 Loaded Datasets:</h3>", unsafe_allow_html=True)
st.write(datasets.keys())  # Dataset keys display

# --- Extracting Patient Data ---
if "patient_ids" in datasets and "times" in datasets and "tokens" in datasets and "patient_data_offsets" in datasets:
    
    # List available patient IDs
    patient_ids = datasets["patient_ids"]
    selected_patient = st.selectbox("🆔 Select Patient ID:", patient_ids)

    # Find patient index
    patient_index = list(patient_ids).index(selected_patient)
    
    st.subheader(f"🆔 Displaying Timeline for Patient: {selected_patient}")

    # Get patient timeline range
    start_idx = datasets["patient_data_offsets"][patient_index]
    end_idx = datasets["patient_data_offsets"][patient_index + 1] if patient_index + 1 < len(datasets["patient_data_offsets"]) else len(datasets["tokens"])

    # Extract data for the patient
    patient_times = datasets["times"][start_idx:end_idx]
    patient_tokens = datasets["tokens"][start_idx:end_idx]
    #decoded_events = [decode([token])[0] for token in patient_tokens]

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

    # Use the function to process patient events correctly
    timeline_df = process_events_with_qx(patient_times, patient_tokens, decode)


    #  # Create DataFrame
    # timeline_df = pd.DataFrame({
    #     "Time": patient_times,
    #     "Event": decoded_events
    # })

    # Convert time values (assuming they are in years) to whole numbers
    timeline_df["Age (Years)"] = np.round(timeline_df["Time"]).astype(int)  # Rounded Age
    #timeline_df["Day"] = np.floor(timeline_df["Time"] * 365).astype(int)  # Convert to Days
    # timeline_df["Cumulative Time (Minutes)"] = (timeline_df["Time"] * 24 * 60).cumsum()
    # timeline_df["Time Diff"] = timeline_df["Cumulative Time (Minutes)"].diff().fillna(0)
    # timeline_df["Hour Group"] = (timeline_df["Time Diff"] > 60).cumsum() + 1  # Start from Hour 1

    # grouped_timeline = timeline_df.groupby("Hour Group").agg({
    # "Cumulative Time (Minutes)": "first",  # Start time of the group
    # "Event": lambda x: ", ".join(x)  # Combine all events in that hour
    # }).reset_index()

    # grouped_timeline["Hour Label"] = "Hour " + grouped_timeline["Hour Group"].astype(str)  # Add labels

    

    # 🔹 **Enhanced Table Appearance**
    st.markdown("<h3 style='font-size:22px; color:##581845;'>📋 Decoded Timeline Data:</h3>", unsafe_allow_html=True)
    st.dataframe(timeline_df, height=400, width=900)  # Bigger table size


    # --- Assign Colors to Unique Events ---
    unique_events = timeline_df["Event"].unique()
    colors = [
        "#FF5733", "#C70039", "#900C3F", "#581845", "#1D8348", "#117A65", "#2874A6",
        "#5B2C6F", "#D4AC0D", "#A93226", "#2471A3", "#D68910", "#28B463", "#7D3C98"
    ]
    event_colors = {event: colors[i % len(colors)] for i, event in enumerate(unique_events)}

    # --- Pagination ---
    events_per_page = 15  # Adjust based on performance
    total_pages = (len(timeline_df) // events_per_page) + (1 if len(timeline_df) % events_per_page else 0)

    # 🔹 **Pagination with Slider**
    # page_number = st.slider("📜 Select Page", min_value=1, max_value=total_pages, value=1) - 1
    # paginated_df = timeline_df.iloc[page_number * events_per_page : (page_number + 1) * events_per_page]


    page_number = st.number_input("📜 Page", min_value=1, max_value=total_pages, step=1) - 1
    paginated_df = timeline_df.iloc[page_number * events_per_page : (page_number + 1) * events_per_page]

    # --- Vertical Timeline Visualization ---
    if not paginated_df.empty:
        st.subheader(f"📈 Timeline Page {page_number + 1}/{total_pages}")

        fig = go.Figure()

        fig.add_annotation(
        x=0.5,  # Center the text horizontally
        y=1.10,  # Position above the highest event
        xref="paper",
        yref="paper",
        text="Age vs Event",  # Title Text
        showarrow=False,
        font=dict(size=20, color="Yellow", family="Arial"),
        align="center"
        )

        # for i, row in paginated_df.iterrows():
        #     fig.add_trace(go.Scatter(
        #         x=[0],  # Keep all events in the center
        #         y=[-i],  # Negative y-values to stack events top-down
        #         mode="markers+text",
        #         marker=dict(size=18, color=event_colors[row["Event"]]),  # Assign event-based color
        #         text=f"{row['Time']} - {row['Event']}",
        #         textposition="middle right",
        #         textfont=dict(size=20),
        #         hoverinfo="text"
        #     ))
        for i, row in paginated_df.iterrows():
          event_name = row["Event"].strip()

            # Extract Q value if present at the end
          q_value = ""
          match = re.search(r"(_Q\d+)$", event_name)  # Look for '_Q' followed by numbers
          if match:
             q_value = match.group(1)  # Extract '_Qx' part
             event_name = event_name.replace(q_value, "").strip()  # Remove Q value from main text
            
            # Truncate long event names but keep Q value
          event_text = (event_name[:20] + "...") if len(event_name) > 20 else event_name
          event_text += f" {q_value}"  # Append Q value next to it

          fig.add_trace(go.Scatter(
            x=[0],  
            y=[-i * 1.5],  # Increase spacing between events
            mode="markers+text",
            marker=dict(size=18, color=event_colors[row["Event"]]),  
            text=f"Age: {round(row['Time'])} | {event_text}",  # Corrected text format
            textposition="middle right",
            textfont=dict(size=16),
            hoverinfo="text",
            hovertext=f"📅 Full Event: {event_name} {q_value}<br>🕒 Age: {round(row['Time'])}"  # Detailed hover info
        ))



        # Layout adjustments
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, autorange="reversed"),  # Reverse for top-down view
            showlegend=False,
            height=600,
            template="plotly_white"
        )

        st.plotly_chart(fig)
    else:
        st.warning("⚠ No events found for the selected patient.")

        

