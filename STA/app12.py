import streamlit as st
import h5py
import pandas as pd
import plotly.graph_objects as go
import os
from ethos.tokenize import Vocabulary  # Replace with actual module

# --- Constants ---
DATA_DIR = "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets"
VOCAB_FILE = os.path.join(DATA_DIR, "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets/mimic_vocab_t763.pkl")

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
st.write("📂 Loaded Datasets:", datasets.keys())  # Debugging

# --- Patient Selection ---
if "patient_ids" in datasets and "age_reference" in datasets:
    patient_df = pd.DataFrame({
        "Patient ID": datasets["patient_ids"],
        "Age Reference": datasets["age_reference"]
    })

    selected_patient = st.selectbox("🆔 Select a Patient", patient_df["Patient ID"].unique())

# --- Extract Patient Timeline ---
if "times" in datasets and "tokens" in datasets and "patient_data_offsets" in datasets:
    # Get patient index
    patient_index = list(datasets["patient_ids"]).index(selected_patient)
    start_idx = datasets["patient_data_offsets"][patient_index]
    end_idx = datasets["patient_data_offsets"][patient_index + 1] if patient_index + 1 < len(datasets["patient_data_offsets"]) else len(datasets["tokens"])

    # Extract data for the patient
    patient_times = datasets["times"][start_idx:end_idx]
    patient_tokens = datasets["tokens"][start_idx:end_idx]
    decoded_events = [decode([token])[0] for token in patient_tokens]

    # Create DataFrame
    timeline_df = pd.DataFrame({
        "Time": patient_times,
        "Event": decoded_events
    })

    st.write("📋 Timeline Data:", timeline_df.head())

    # --- Visualizing the Timeline ---
    if not timeline_df.empty:
        st.subheader("📈 Horizontal Timeline Visualization")

        colors = ["#FF5733", "#C70039", "#900C3F", "#581845", "#1D8348", "#117A65", "#2874A6", "#5B2C6F"]
        event_colors = {event: colors[i % len(colors)] for i, event in enumerate(timeline_df["Event"].unique())}

        fig = go.Figure()

        for i, row in timeline_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Time"]],
                y=[1],  # Keep all events on the same level
                mode="markers+text",
                marker=dict(size=20, color=event_colors[row["Event"]], symbol="square"),
                text=row["Event"],
                textposition="top center",
                hoverinfo="text+x"
            ))

        # Layout adjustments
        fig.update_layout(
            xaxis_title="Time (Hours)",
            yaxis=dict(visible=False),
            xaxis=dict(showgrid=False, zeroline=False),
            showlegend=False,
            height=300,
            template="plotly_white"
        )

        st.plotly_chart(fig)
    else:
        st.warning("⚠ No events found for the selected patient.")
