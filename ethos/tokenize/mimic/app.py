import pandas as pd

from ..base import SimpleData
from ..translation_base import Translation
from ...constants import PROJECT_DATA


class DrugTranslation(Translation):
    def __init__(self, data_prop):
        self.data_prop = data_prop
        self._name_to_code: pd.Series = None
        self._code_to_name: dict = None

    def _create_name_to_code_translation(self) -> pd.Series:
        drug_to_gsn = SimpleData(
            "hosp/prescriptions",
            self.data_prop,
            use_cols=["drug", "gsn"],
            no_id_ok=True,
            dtype=str,
        ).df
        drug_to_gsn.gsn = drug_to_gsn.gsn.str.strip()
        drug_to_gsn.gsn = drug_to_gsn.gsn.str.split(" ")
        drug_to_gsn = drug_to_gsn.explode("gsn")
        gsn_to_atc = self._load_gsn_to_atc().set_index("gsn").atc
        self._name_to_code = (
            drug_to_gsn.set_index("drug")
            .gsn.map(gsn_to_atc, na_action="ignore")
            .reset_index()
            .drop_duplicates()
            .dropna()
            .rename({"gsn": "atc_code"}, axis=1)
            .set_index("drug")
            .atc_code
        )
        return self._name_to_code

    def _create_code_to_name_translation(self) -> dict:
        if self._name_to_code is None:
            self._name_to_code = self._create_name_to_code_translation()
        self._code_to_name = self._name_to_code.reset_index().set_index("atc_code").drug.to_dict()
        return self._code_to_name

    @staticmethod
    def _load_gsn_to_atc() -> pd.DataFrame:
        return pd.read_csv(
            PROJECT_DATA / "gsn_atc_ndc_mapping.csv.gz", usecols=["gsn", "atc"], dtype=str
        )


# Initialize translation
data_prop = {}  # Replace with actual data properties
drug_translation = DrugTranslation(data_prop)

# Generate mappings
name_to_code = drug_translation._create_name_to_code_translation()
code_to_name = drug_translation._create_code_to_name_translation()

# Display mappings
print("\n📌 Drug → ATC Mapping:")
print(name_to_code.reset_index().rename(columns={"drug": "Drug Name", "atc_code": "ATC Code"}))

print("\n📌 ATC → Drug Mapping:")
print(pd.DataFrame(list(code_to_name.items()), columns=["ATC Code", "Drug Name"]))

# CLI-based search
while True:
    print("\n🔹 Search Options:")
    print("1. Search Drug → ATC")
    print("2. Search ATC → Drug")
    print("3. Exit")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        drug_query = input("Enter Drug Name: ").strip()
        atc_code = name_to_code.get(drug_query, "Not Found")
        print(f"🔍 ATC Code for {drug_query}: {atc_code}")

    elif choice == "2":
        atc_query = input("Enter ATC Code: ").strip()
        drug_name = code_to_name.get(atc_query, "Not Found")
        print(f"🔍 Drug Name for {atc_query}: {drug_name}")

    elif choice == "3":
        print("✅ Exiting program.")
        break

    else:
        print("❌ Invalid choice. Please try again.")
