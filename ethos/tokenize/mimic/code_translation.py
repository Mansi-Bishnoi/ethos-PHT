import pandas as pd

from ..base import SimpleData
from ..translation_base import Translation
from ...constants import PROJECT_DATA



class DrugTranslation(Translation):
    def __init__(self, data_prop):
        self.data_prop = data_prop
        self._name_to_code: pd.Series = None

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
        
        # Load GSN to Drug Name mapping
        self._gsn_to_drug = (
            drug_to_gsn[['gsn', 'drug']]
            .dropna()
            .drop_duplicates()
            .set_index("gsn")
            .to_dict()["drug"]
        )
        # Mapping Drug → GSN → ATC
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
        # DEBUG PRINT: Check Drug → ATC Mapping
        print("Sample Drug to ATC Mapping:", self._name_to_code.head())
        return self._name_to_code

    def _create_code_to_name_translation(self) -> dict:
        if self._name_to_code is None:
            self._name_to_code = self._create_name_to_code_translation()
        return self._name_to_code.reset_index().set_index("atc_code").drug.to_dict()
    
    def get_gsn_to_drug_mapping(self) -> dict:
        """Returns the GSN → Drug Name mapping."""
        if self._gsn_to_drug is None:
            self._create_name_to_code_translation()  # Ensure mapping is generated
        
        # DEBUG PRINT: Check if GSN to Drug Name mapping is working
        print("Sample GSN to Drug Mapping:", list(self._gsn_to_drug.items())[:10])

        return self._gsn_to_drug

    @staticmethod
    def _load_gsn_to_atc() -> pd.Series:
        return pd.read_csv(
            PROJECT_DATA / "gsn_atc_ndc_mapping.csv.gz", usecols=["gsn", "atc"], dtype=str
        )


class _IcdTranslation(Translation):
    def __init__(self, data_prop, data_name):
        self.data_prop = data_prop
        self.data_name = data_name

    def _create_name_to_code_translation(self):
        df = self._load_icd_data()
        return (
            df.loc[df.icd_version == 10]
            .groupby("long_title")
            .icd_code.agg(lambda values: max(values, key=len))
            .to_dict()
        )

    def _create_code_to_name_translation(self):
        df = self._load_icd_data()
        return (
            df.loc[df.icd_version == 10]
            .groupby("icd_code")
            .long_title.agg(lambda values: min(values, key=len))
            .to_dict()
        )

    def _load_icd_data(self):
        return SimpleData(
            self.data_name,
            self.data_prop,
            use_cols=["icd_code", "long_title", "icd_version"],
            no_id_ok=True,
        ).df


class IcdCmTranslation(_IcdTranslation):
    def __init__(self, data_prop):
        super().__init__(data_prop, "hosp/d_icd_diagnoses")


class IcdPcsTranslation(_IcdTranslation):
    def __init__(self, data_prop):
        super().__init__(data_prop, "hosp/d_icd_procedures")
