
#import libraries
import pandas as pd
from chembl_webresource_client.new_client import new_client
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import warnings
warnings.filterwarnings("ignore")
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle
import time
from PIL import Image
from io import BytesIO

# intro & formatting

st.markdown(
    """
<style>
.reportview-container .markdown-text-container {
    
    font-family: IBM Plex Sans;
}
.sidebar .sidebar-content {
    background-image: url("https://i.pinimg.com/originals/53/3e/f4/533ef47e9ffb0f69f644e307696931b5.jpg");
    color: white;
}
.Widget>label {
    color: white;
    font-family: monospace;
}
[class^="st-b"]  {
    color: white;
    font-family: monospace;
}
.st-bb {
    background-color: #d19da6;
}
.st-at {
    background-color: #3d393a;
}
footer {
    font-family: monospace;

}
.reportview-container .main footer, .reportview-container .main footer a {
    color: #013d29;
}
header .decoration {
    background-image: url("https://img2.mahoneswallpapershop.com/prodimage/ProductImage/560/aecd25f8-4822-42ca-85f7-62d63cd41fc3.jpg");
}

</style>
""",
    unsafe_allow_html=True,
)

st.title('💊 Drug Discovery Pipeline')
st.text('by Corey J Sinnott')
st.subheader('This app creates a model from every available drug in the \
              ChEMBL database, and will return a Minimum Inhibitory \
              Concentration (**MIC**) prediction for your compound\
              in **nM**.')

# ------------ dataframe creation ------------ #

def target_search(user_enter_organism):
    """
    Searches the ChEMBL database for a target organism; 
    filters for available relevant data.

    Args: User input of a bacteria name.

    Returns: cleaned dataframe of necessary information.
    """
    #user_enter_organism = st.text_input('organism name') #outside of function? input as entry?
    st.spinner()
    with st.spinner(text='Target Search In-Progress'):
        
        target = new_client.target
        time.sleep(3)
        target_search = target.search(user_enter_organism)
        st.write('- organism obtained ✔️')
        time.sleep(3)
        target_df = pd.DataFrame.from_dict(target_search)
        st.write('- drugs obtained ✔️')
        time.sleep(3)
        select_target = target_df.target_chembl_id[0]
        time.sleep(3)
        activity = new_client.activity
        time.sleep(3)
        res = activity.filter(target_chembl_id = select_target).filter(standard_type=['MIC'])
        time.sleep(1)
        df_test = pd.DataFrame.from_dict(res)
        time.sleep(1)
        df_test['standard_value'] = df_test['standard_value'].astype(float)
        time.sleep(1)
        df_trim = df_test.dropna(subset = ['standard_value', 'canonical_smiles'])
        df = df_trim[['canonical_smiles', 'standard_value']].drop_duplicates(subset = ['canonical_smiles'])
        st.success('Target search complete')
    return df


# ------------ feature engineering ------------ #

def bioactivity_rater(df):
    #paste in rater if needed
    #will require a percentile calc
    pass

def add_mol_feats_v2(df, df_col):
    """
    Creates a new dataframe with four new molecular descriptors: molecular
    weight, log P, proton donors, and proton acceptors.
    
    Args: dataframe column containing SMILES
    
    Returns: new dataframe of molecular features
    """

    st.spinner()
    with st.spinner(text='Featurization In-Progress'):

        mol_wt   = [round(Descriptors.MolWt(Chem.MolFromSmiles(i)), 3) for i in df_col]
        st.write('- molecular weight added ✔️')
        log_p    = [Descriptors.MolLogP(Chem.MolFromSmiles(i)) for i in df_col]
        st.write('- log-p added ✔️')
        H_donors = [Lipinski.NumHDonors(Chem.MolFromSmiles(i)) for i in df_col]
        st.write('- proton donors added ✔️')
        H_accept = [Lipinski.NumHAcceptors(Chem.MolFromSmiles(i)) for i in df_col]
        st.write('- proton acceptors added ✔️')
        # can add more features as needed
        
        mol_array = np.array([mol_wt,
                            log_p,
                            H_donors,
                            H_accept])
        
        mol_array_trans = np.transpose(mol_array)
        
        mol_df = pd.DataFrame(data = mol_array_trans, 
                            columns = [
                                'mol_wt',
                                'log_p',
                                'proton_donors',
                                'proton_acceptors'
                                            ])
        
        df_2 = pd.concat([df.reset_index(), mol_df], axis = 1)
        st.success('Featurization complete')

    return df_2 #O(n^4)

def finger_printer(df):
    """
    Creates a vectorized representation of SMILES using Morgan's Fingerprint Algorithm
    
    Args: Dataframe

    Returns: Dataframe with fingerprints
    """
    col = df['canonical_smiles']

    ECFP6 = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i), 3) for i in col] 
    ECFP6_list = [list(i) for i in ECFP6]
    ECFP6_names = [f'Bit_{i}' for i in range(2048)]
    ECFP6_df = pd.DataFrame(ECFP6_list, index = df.index, columns = ECFP6_names)
    
    st.write('- Morgan fingerprinting complete ✔️')

    df_3 = pd.concat([df.reset_index(drop = True), ECFP6_df], axis = 1)
    
    return df_3 #O(n^3)

def pMIC(df):
    """
    Converts standard value (MIC) to a -log10 standardized value for 
    better distribution.

    Args: df

    Returns: df with new columns
    """
    st.write('standardizing target values with -log10')

    df['pMIC'] = df['standard_value'].map(lambda x: -np.log10(x * (10**-9)))

    st.write('- target values standardized')

    return df_4

# ------------ modeling ------------ #    

def regression_and_eval(df):
    """
    Takes in x and y variables, and fits a regression model.

    Args: X, y, model

    Returns: trained model, X_train, X_test, y_train, y_test, y_pred
    """
    df['pMIC'] = df['standard_value'].map(lambda x: -np.log10(x * (10**-9)))

    st.subheader('Analyzing')
    
    X = df.drop(columns = ['canonical_smiles',
                           'standard_value',
                           'pMIC', 'index'])
    #y = df['pMIC']
    y = df['standard_value']

    ss = StandardScaler()
    num_cols = ['mol_wt', 'log_p', 'proton_donors', 'proton_acceptors']
    X[num_cols] = ss.fit_transform(X[num_cols])

    model = RandomForestRegressor()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2219)
    null_y = np.full_like(y_train, y.mean())

    st.spinner()
    with st.spinner(text='Random Forest regression in-progress ...'):

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success('Modeling complete')

    null_MSE = mean_squared_error(y_train, null_y)
    MSE = mean_squared_error(y_test, y_pred)

    f'Null MSE = {np.round(mean_squared_error(y_train, null_y), 3)}'
    f'MSE   = {np.round(mean_squared_error(y_test, y_pred), 3)}'
    f'RMSE  = {np.round(mean_squared_error(y_test, y_pred, squared = False), 3)}'
    f'MAE   = {np.round(mean_absolute_error(y_test, y_pred), 3)}'
    f'r^2   = {np.round(r2_score(y_test, y_pred), 3)}'
    #f'MSE% greater than null = {(np.abs(np.round(((null_MSE - MSE) / null_MSE), 3)))*100}%'

    return model

# ------------------ input processing ------------------------ #

def prediction_prep(smile):
    """
    Prepares the SMILE inputted by the user for analysis.
    
    Args: SMILE.
    
    Returns: SMILE with appropriate features and vectorization.
    """
    st.spinner()
    with st.spinner(text='Featurizing your molecule and making prediction ...'):
        ss = StandardScaler()

        #adding Lipinkis
        mol_wt   = round(Descriptors.MolWt(Chem.MolFromSmiles(smile)), 3)
        st.write(f'- molecular weight added [{mol_wt}]')
        log_p    = round(Descriptors.MolLogP(Chem.MolFromSmiles(smile)), 3)
        st.write(f'- log-p added [{log_p}] ✔️')
        H_donors = Lipinski.NumHDonors(Chem.MolFromSmiles(smile))
        st.write(f'- proton donors added [{H_donors}]')
        H_accept = Lipinski.NumHAcceptors(Chem.MolFromSmiles(smile))
        st.write(f'- proton acceptors added [{H_accept}]')
        
        feats_df = pd.DataFrame([mol_wt, log_p, H_donors, H_accept])
        feats_df = pd.DataFrame(ss.fit_transform(feats_df))
        feats_df = feats_df.T
        feats_df = feats_df.rename(columns = {0 : 'Mol Wt.', 
                                              1 : 'log P',
                                              2 : 'Proton Donors',
                                              3 : 'Proton Acceptors' 
                                                    })
        #vectorizing
        prints = list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 3))
        prints_df = pd.DataFrame(prints).T

        X_user = pd.concat([feats_df, prints_df], axis = 1)

        st.write('- Morgan fingerprinting complete')
        st.success('Featurization complete')
    return X_user

# -------------------- interface --------------------------- #

with st.sidebar.header('Enter an organism 🦠 and a SMILE 🧪 to get started.'):
    user_enter_organism = st.sidebar.text_input('🦠 Organism:')
    user_smile = st.sidebar.text_input('🧪 SMILE:')
    button = st.button('Submit')

if button:
    df = target_search(user_enter_organism)
    df_2 = add_mol_feats_v2(df, df['canonical_smiles'])
    df_3 = finger_printer(df_2)
    #df_4 = pMIC(df_3)
    model = regression_and_eval(df_3)
    time.sleep(1)
    st.subheader('Making prediction')
    X_user = prediction_prep(user_smile)
    time.sleep(1)
    prediction = model.predict(X_user)
    # displays molecule image
    img = Draw.MolToImage(Chem.MolFromSmiles(user_smile))
    bio = BytesIO()
    img.save(bio, format='png')
    st.image(img)
    st.write(user_smile)
    st.subheader(f'Predicted MIC: {round((prediction[0]), 2)}nM') 

# re-running with trained model
    next_button = st.button('Make another prediction')
    while next_button:
        user_smile = st.text_input('Next SMILE')
        st.subheader('Making prediction')
        X_user = prediction_prep(user_smile)
        time.sleep(1)
        prediction = model.predict(X_user)
        # displays molecule image
        img = Draw.MolToImage(Chem.MolFromSmiles(user_smile))
        bio = BytesIO()
        img.save(bio, format='png')
        st.image(img)
        st.write(user_smile)
        st.subheader(f'Predicted MIC: {round((prediction[0]), 2)}nM') 