import mols2grid
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components

st.markdown("""
# Malaria Bioactivity: EC50 Analysis for Inhibiting Malaria

This app takes a dataset of molecules (SMILES) and allows us to visualize them.
Specifically, we can track the bioactivity of drug molecules known to inhibit malaria.
EC50 represents the concentration of a drug at a stable state for half the maximum effect.

**Credits**
- App built in `Python` + `Streamlit` by [Saamahn Mahjouri](https://www.linkedin.com/in/saamahnmahjouri/)
---
""")

@st.cache(allow_output_mutation=True)
def download_dataset():
    """Loads once then cached for subsequent runs"""
    df = pd.read_csv(
        "https://raw.githubusercontent.com/OpenDrugAI/AttentiveFP/master/data/malaria-processed.csv", 
        names = ["Loge EC50", "smiles"]
    ).dropna()
    return df

# Copy the dataset so any changes are not applied to the original cached version
df = download_dataset().copy()

# EC50 - concentration of drug at a stable state inducing half of maximum effect
ec50_cutoff = st.slider(
    label="Show compounds below the EC50 cutoff:",
    min_value=-7.0,
    max_value=3.0,
    value=0.5,
    step=0.1,
)

df_result = df[df["Loge EC50"] < ec50_cutoff]

st.write(df_result)

raw_html = mols2grid.display(df_result)._repr_html_()
components.html(raw_html, width=900, height=900, scrolling=True)