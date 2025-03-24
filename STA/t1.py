import h5py
import os

DATA_DIR = "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets"
selected_token = "mimic_test_timelines_p10.hdf5"  # Replace with the actual file name

token_path = os.path.join(DATA_DIR, selected_token)

with h5py.File(token_path, "r") as f:
    print("Available datasets in the file:")
    print(list(f.keys()))  # List all available datasets


file_path = "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets/mimic_test_timelines_p10.hdf5"  # Replace with your file path

def print_hdf5_structure(file_path):
    with h5py.File(file_path, "r") as f:
        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")

        f.visititems(print_attrs)

print_hdf5_structure(file_path)


with h5py.File(file_path, "r") as f:
    print("📂 HDF5 File Contents:")
    
    # Print dataset names
    for key in f.keys():
        print(f"🔹 {key}: {f[key].shape} {f[key].dtype}")

    # Read specific datasets
    age_reference = f["age_reference"][:] if "age_reference" in f else None
    patient_context = f["patient_context"][:] if "patient_context" in f else None
    patient_data_offsets = f["patient_data_offsets"][:] if "patient_data_offsets" in f else None
    patient_ids = f["patient_ids"][:] if "patient_ids" in f else None
    times = f["times"][:] if "times" in f else None
    tokens = f["tokens"][:] if "tokens" in f else None

    # Display some data
    if age_reference is not None:
        print("\n🩺 Age Reference (First 10 Patients):", age_reference[:10])

    if patient_ids is not None:
        print("\n🆔 Patient IDs (First 10 IDs):", patient_ids[:10])

    if times is not None and tokens is not None:
        print("\n⏳ Timeline Sample:")
        for i in range(10):  # First 10 entries
            print(f"Time: {times[i]}, Token: {tokens[i]}")

from ethos.tokenize import Vocabulary  # Replace with actual module
import h5py

# Load Vocabulary
vocab = Vocabulary("/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets/mimic_vocab_t763.pkl")  # Ensure the correct path
decode = vocab.decode


with h5py.File(file_path, "r") as f:
    patient_ids = f["patient_ids"][:]
    times = f["times"][:]
    tokens = f["tokens"][:]

    # Decode first 10 tokens
    decoded_tokens = [decode([token]) for token in tokens[:10]]

    print("\n🔹 Decoded Timeline Sample:")
    for i, (t, token) in enumerate(zip(times[:10], decoded_tokens)):
        print(f"Time: {t:.2f}, Event: {token}")


