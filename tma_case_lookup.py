import pandas as pd


def get_original_tma_case(tma_case_anonymized):
    """
    Given an anonymized TMA case ID, return the original TMA case.
    """
    patient_meta_df = pd.read_csv('data/patient_meta_anonymized.csv')
    tma_case = patient_meta_df.loc[patient_meta_df["tma_case_anonymized"] == tma_case_anonymized, "tma_case"].values
    if len(tma_case) == 0:
        return None
    return tma_case[0]
