import h5py
import pickle
import numpy as np
from datetime import datetime, timedelta
from bisect import bisect
import pandas as pd

# Load HDF5 file (timeline data)
def load_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        data = {
            "age_reference": f["age_reference"][:],  # Directly loading here
            "patient_context": f["patient_context"][:],
            "patient_data_offsets": f["patient_data_offsets"][:],
            "patient_ids": f["patient_ids"][:],
            "times": f["times"][:],
            "tokens": f["tokens"][:]
        }
    return data

# Load vocab file (token ID -> event name mapping)
def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab["itos"]  # itos[index] gives the corresponding text

def convert_time(patient_idx, event_time, age_reference):
    """
    Converts event time to an absolute date based on patient's age reference.
    
    Args:
    patient_idx (int): Index of the patient in the dataset.
    event_time (float): Event time in relation to age reference.
    age_reference (dict): Dictionary mapping patient index to reference age.

    Returns:
    str: Formatted event date.
    """
    if patient_idx not in age_reference:
        return "Unknown Date"

    birth_year = age_reference[patient_idx]  # Get patient's birth year

    # Calculate event year based on birth year
    event_year = birth_year + event_time / 365.25  # Convert days to years

    return f"{event_year:.2f}"  # Return formatted year


def process_timeline_events(times, tokens, decode):
    """
    Processes timeline events and calculates time passed between them.
    :param times: List of timestamps for events (in years)
    :param tokens: List of tokenized events
    :param decode: Function to decode tokens into readable events
    :return: Pandas DataFrame with events and time differences
    """
    events = []
    
    for i in range(len(tokens)):
        decoded_token = decode([tokens[i]])[0]

        # Calculate time difference
        if i == 0:
            time_diff = 0  # No time passed for the first event
        else:
            time_diff = times[i] - times[0]  # Difference in years

        # Convert time difference to days, hours, minutes
        time_diff_days = round(time_diff * 365)
        time_diff_hours = round(time_diff * 365 * 24)
        time_diff_minutes = round(time_diff * 365 * 24 * 60)

         # Dynamically set the time display format
        if time_diff_minutes < 60:
                time_display = f"{time_diff_minutes} minutes"
        elif time_diff_hours < 24:
                time_display = f"{time_diff_hours} hours"
        else:
                time_display = f"{time_diff_days} days"

        events.append({
            "Event": decoded_token,
            "Token Mapping": tokens[i],
            "Time Passed (Days)": time_diff_days,
            "Time Passed (Hours)": time_diff_hours,
            "Time Passed (Minutes)": time_diff_minutes
        })

    return pd.DataFrame(events)

def get_patient_timeline(patient_id, data, decode):
    """
    Prints the timeline of a given patient by extracting relevant event details.
    
    Args:
    patient_id (int): ID of the patient to fetch timeline for.
    data (dict): Loaded HDF5 dataset containing patient events.
    decode (function): Function to decode tokens into readable events.
    
    Returns:
    pd.DataFrame: Timeline of the patient's medical events.
    """
    patient_ids = data["patient_ids"]
    offsets = data["patient_data_offsets"]
    
    # Find index of the given patient_id
    try:
        patient_idx = np.where(patient_ids == patient_id)[0][0]
    except IndexError:
        print(f"Patient ID {patient_id} not found in dataset.")
        return None

    # Get start and end indices for this patient's events
    start_idx = offsets[patient_idx]
    end_idx = offsets[patient_idx + 1] if patient_idx + 1 < len(offsets) else len(data["tokens"])

    # Extract relevant times and tokens
    patient_times = data["times"][start_idx:end_idx]
    patient_tokens = data["tokens"][start_idx:end_idx]
    
    # Process and return the timeline
    return process_timeline_events(patient_times, patient_tokens, decode)



# def process_events(data, itos):
#     items = []
#     last_patient_idx = None  # Track last processed patient

#     for i in range(len(data["times"])):
#         token_id = data["tokens"][i]
#         event_desc = itos.get(token_id, f"Event {token_id}")

#         patient_idx = bisect(data["patient_data_offsets"], i) - 1
#         event_date = convert_time(patient_idx, data["times"][i], data["age_reference"])

#         # Print birth year only once per patient
#         if patient_idx != last_patient_idx:
#             birth_year = data["age_reference"].get(patient_idx, "Unknown Birth Year")
#             print(f"\nPatient {patient_idx}: Birth Year ~ {birth_year}")  
#             last_patient_idx = patient_idx  # Update last processed patient

#         print(f"  Event {i}: Token {token_id} -> '{event_desc}' on {event_date}")

#         items.append({
#             "id": i,
#             "content": event_desc,
#             "start": event_date
#         })
    
#     return items



