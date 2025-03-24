import streamlit as st
import h5py
import pandas as pd
import plotly.graph_objects as go
import os
from ethos.tokenize import Vocabulary  # Replace with actual module

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
st.write("📂 Loaded Datasets:", datasets.keys())  # Debugging

# --- Extracting Single Patient Data ---
if "patient_ids" in datasets and "times" in datasets and "tokens" in datasets and "patient_data_offsets" in datasets:
    
    selected_patient = datasets["patient_ids"][0]  # Only selecting 1st patient
    st.subheader(f"🆔 Patient: {selected_patient}")

    # Get patient index (always 0 as we're selecting the first)
    patient_index = 0
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

    st.write("📋 Decoded Timeline Data:", timeline_df.head())

    # --- Assign Colors to Unique Events ---
    unique_events = timeline_df["Event"].unique()
    colors = [
        "#FF5733", "#C70039", "#900C3F", "#581845", "#1D8348", "#117A65", "#2874A6",
        "#5B2C6F", "#D4AC0D", "#A93226", "#2471A3", "#D68910", "#28B463", "#7D3C98"
    ]
    event_colors = {event: colors[i % len(colors)] for i, event in enumerate(unique_events)}

    # --- Pagination ---
    events_per_page = 10  # Adjust based on performance
    total_pages = (len(timeline_df) // events_per_page) + (1 if len(timeline_df) % events_per_page else 0)

    page_number = st.number_input("📜 Page", min_value=1, max_value=total_pages, step=1) - 1
    paginated_df = timeline_df.iloc[page_number * events_per_page : (page_number + 1) * events_per_page]

    # --- Vertical Timeline Visualization ---
    if not paginated_df.empty:
        st.subheader(f"📈 Timeline Page {page_number + 1}/{total_pages}")

        fig = go.Figure()

        for i, row in paginated_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[0],  # Keep all events in the center
                y=[-i],  # Negative y-values to stack events top-down
                mode="markers+text",
                marker=dict(size=15, color=event_colors[row["Event"]]),  # Assign event-based color
                text=f"{row['Time']} - {row['Event']}",
                textposition="middle right",
                hoverinfo="text"
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
