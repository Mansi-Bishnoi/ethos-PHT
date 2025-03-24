import pandas as pd

# Load ICD-9 to ICD-10 mapping file
icd_mapping = pd.read_csv("/home/shtlp_0094/workspace/ethos-paper/ethos/data/icd_cm_9_to_10_mapping.csv")  # Replace with actual file name
# Load ICD-9 descriptions
icd_descriptions = pd.read_csv("/home/shtlp_0094/workspace/ethos-paper/ethos/tokenize/d_icd_diagnoses.csv")

# Merge to get ICD-10 codes
merged_df = icd_descriptions.merge(icd_mapping, left_on="icd_code", right_on="icd_9", how="left")

# Load ICD-10 descriptions
icd10_descriptions = pd.read_csv("/home/shtlp_0094/workspace/ethos-paper/ethos/data/icd10cm-order-Jan-2021.csv")  # Replace with actual file

# Merge to get ICD-10 descriptions
final_df = merged_df.merge(icd10_descriptions, left_on="icd_code", right_on="code", how="left")

# Save the final mapping with descriptions
final_df.to_csv("icd9_to_icd10_with_descriptions.csv", index=False)

print("Final CSV saved as icd9_to_icd10_with_descriptions.csv")

