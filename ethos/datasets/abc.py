from timeline import load_hdf5, load_vocab, process_timeline_events, get_patient_timeline
import pandas as pd
import numpy as np 

# File paths
HDF5_FILE = "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets/mimic_test_timelines_p10.hdf5"
VOCAB_FILE = "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets/mimic_vocab_t763.pkl"

# Load data
data = load_hdf5(HDF5_FILE)
itos = load_vocab(VOCAB_FILE)

# Extract `times` and `tokens` from data
times = data["times"]
tokens = data["tokens"]

def decode(token_list):
    return [itos[token] if token in itos else f"UNKNOWN_{token}" for token in token_list]

# Get timeline for a specific patient
patient_id = 10039997
patient_timeline = get_patient_timeline(patient_id, data, decode)

# Display the timeline if found
if patient_timeline is not None:
    print(f"Timeline for Patient {patient_id}:\n")
    print(patient_timeline)

patient_timeline.to_csv(f"patient_{patient_id}_timeline.csv", index=False)
print(f"Timeline saved to patient_{patient_id}_timeline.csv")
