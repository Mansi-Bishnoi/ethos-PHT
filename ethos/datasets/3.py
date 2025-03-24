import pandas as pd

def get_combined_atc_code(csv_file, atc_1, atc_2, atc_3):
    """
    Function to extract ATC code based on given three tokens.

    Parameters:
    - csv_file: Path to CSV file containing 'atc_name' and 'atc_code'.
    - atc_1: First token (e.g., "ATC_analgesics").
    - atc_2: Second token (e.g., "ATC_4_B").
    - atc_3: Third token (e.g., "ATC_SUFFIX_E01").

    Returns:
    - Final combined ATC code and its corresponding name.
    """

    # Extract values from tokens
    a = atc_1.split("_")[1]  # Extract "analgesics"
    b = atc_2.split("_")[-1]  # Extract "B"
    suffix = atc_3.split("_")[-1]  # Extract "E01"

    # Load CSV
    df = pd.read_csv('/home/shtlp_0094/workspace/ethos-paper/ethos/data/atc_coding.csv')

    # Find corresponding code for 'a' (analgesics)
    row = df[df['atc_name'].str.lower() == a.lower()]
    
    if row.empty:
        return f"Error: No ATC code found for {a} in CSV"

    atc_code = row['atc_code'].values[0]  # Get corresponding ATC code

    # Create final combined ATC code
    final_atc_code = f"{atc_code}{b}{suffix}"

    # Find the corresponding name for the final ATC code
    final_row = df[df['atc_code'] == final_atc_code]

    if final_row.empty:
        return final_atc_code, "No matching ATC name found in CSV"
    
    final_atc_name = final_row['atc_name'].values[0]

    return {
        "part_1": { "token_value": a, "mapped_code": atc_code},
        "part_2": { "token_value": b, "appended_value": b},
        "part_3": { "token_value": suffix, "appended_suffix": suffix},
        "final_code": final_atc_code,
        "final_name": final_atc_name
    }
map_df = pd.read_csv('/home/shtlp_0094/workspace/ethos-paper/ethos/data/icd_pcs_9_to_10_mapping.csv')

# Debugging version with extra prints
def get_icd_9_from_icd_pcs(map_df, event):
    """
    Maps ICD PCS token to corresponding ICD 9 code.

    Parameters:
    - map_df: DataFrame with 'icd_9' and 'icd_10' (ICD PCS mapping).
    - event: Complete token string (e.g., 'ICD_PCS 3VJ4CZ').

    Returns:
    - Corresponding ICD 9 code if found, else None.
    """
    print(f"Received event: {event}")
    
    # Split and validate format
    tokens = event.split(" ")
    if len(tokens) == 2 and tokens[0].startswith("ICD_PCS"):
        icd_pcs_code = tokens[1].strip().upper()  # Clean and standardize PCS code
        print(f"Extracted PCS code: {icd_pcs_code}")

        # Check if PCS code exists in map_df
        row = map_df[map_df['icd_10'] == icd_pcs_code]
        if not row.empty:
            icd_9_code = row.iloc[0]['icd_9']
            print(f"Match found! ICD 9 Code: {icd_9_code}")
            return icd_9_code
        else:
            print(f"No match found for ICD PCS code: {icd_pcs_code}")
    
    print("Invalid format or no match found.")
    return None

# Test with ICD_PCS formatted event
event_icd_pcs = "ICD_PCS 03VJ4CZ"
icd_9_code = get_icd_9_from_icd_pcs(map_df, event_icd_pcs)

if icd_9_code:
    print(f"✅ ICD 9 Code: {icd_9_code}")
else:
    print("❌ No matching ICD 9 code found.")