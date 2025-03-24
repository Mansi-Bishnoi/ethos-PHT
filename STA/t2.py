import h5py
import os
from ethos.tokenize import Vocabulary  # Replace with actual module

# Define the data directory and file name
DATA_DIR = "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets"
selected_token = "mimic_test_timelines_p10.hdf5"  # Replace with the actual file name
file_path = os.path.join(DATA_DIR, selected_token)

# Load Vocabulary for Decoding
vocab = Vocabulary(os.path.join(DATA_DIR, "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets/mimic_vocab_t763.pkl"))  # Ensure correct path
decode = vocab.decode

# Function to print HDF5 structure
def print_hdf5_structure(file_path):
    with h5py.File(file_path, "r") as f:
        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")

        f.visititems(print_attrs)

print("\n🔍 **HDF5 File Structure:**")
print_hdf5_structure(file_path)

# Read HDF5 File
with h5py.File(file_path, "r") as f:
    print("\n📂 **HDF5 File Contents:**")

    # Print available datasets
    for key in f.keys():
        print(f"🔹 {key}: {f[key].shape} {f[key].dtype}")

    # Read datasets (handling missing keys)
    age_reference = f.get("age_reference")
    patient_context = f.get("patient_context")
    patient_data_offsets = f.get("patient_data_offsets")
    patient_ids = f.get("patient_ids")
    times = f.get("times")
    tokens = f.get("tokens")

    # Display patient-related information
    if age_reference is not None:
        print("\n🩺 **Age Reference (All Patients):**", age_reference[:])

    if patient_ids is not None:
        print("\n🆔 **Patient IDs (All IDs):**", patient_ids[:])

    # Process timeline and decode all tokens
    if times is not None and tokens is not None:
        decoded_tokens = [decode([token]) for token in tokens]  # Decode all tokens
        print("\n⏳ **Decoded Timeline (Full Data):**")
        for t, token in zip(times, decoded_tokens):
            print(f"Time: {t:.2f}, Event: {token}")
