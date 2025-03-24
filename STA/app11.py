import h5py
import os
from ethos.tokenize import Vocabulary

# # Open the HDF5 file
# with h5py.File('/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets/mimic_test_timelines_p10.hdf5', 'r') as hdf:
#     # List all groups and datasets
#     print("Keys: ", list(hdf.keys()))

#     # Access a specific dataset
#     dataset_name = list(hdf.keys())[2]  # Get the first dataset name
#     data = hdf[dataset_name][:]
    
#     print("Dataset: ", dataset_name)
#     print("Data: ", data)

DATA_DIR = "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets"
VOCAB_FILE = os.path.join(DATA_DIR, "mimic_vocab_t763.pkl")  # Adjust if needed


# Load Vocabulary
vocab = Vocabulary(VOCAB_FILE)
decode = vocab.decode
print(Vocabulary)


