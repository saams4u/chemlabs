#### Antibiotic Drug Discovery
#### Corey J Sinnott
   
## Executive Summary
   
This report was commissioned to develop a robust, fast, and reproducible drug discovery pipeline for new antibiotics, in an effort to combat antibiotic resistance. After in-depth analysis, conclusions and recommendations will be presented.
   
Data was obtained from the following source:
- Comprehensive Antibiotic Resistance Database via CARD CLI interface: 
 - https://card.mcmaster.ca
- ChEMBL via Python client library: 
 - https://www.ebi.ac.uk/chembl/  


In the coming years, it will become crucial to develop drug discovery pipelines using machine learning to expedite the search for new antibiotics. Towse et al. estimate the cost of developing a new antibiotic via the traditional research pipeline could cost over 1.7B USD, and take over 10 years. This makes for a bleak future, where more strains of bacteria become resistant to traditional antibiotics each day, and pharmaceutical companies do not see a cost-benefit to researching new drugs. Machine learning based drug discovery aims to take several years and hundreds of millions of USD off of the current estimates. With this goal in mind, one bacteria, Acinetobacter baumannii, was chosen, and a series of drugs with a recorded efficacy were aggregated using the ChEMBL web-resources client. These results were not filtered for Lipinski's rule of 5, as current research indicates these rules do not apply to antibiotics. The entire pipeline was optimized to be adaptable to any organism search. 

After obtaining and exploring data for over 4000 compounds, functions were engineered to classify and featurize the molecules. First, each molecule was rated as "active", "intermediate," or "inactive" based on the recorded standard value - a nM scale representation of potency. Next, features were engineered using the RDKit library. Columns were added for logP, molecular weight, number of proton donors, and number of proton acceptors. The functions were designed for flexibility in adding additional features. The final featurization step used the RDKit Morgan fingerprint method to binarize each molecule based on a combination of atoms and functional groups. This method of vectorization will allow us to look at the most important molecular fragments when performing feature extraction.

After featurizing each compound, each numerical feature was scaled using SKLearn's StandardScaler, and LazyPredict was used to search for the best potential models. HistGradient Boosting regression and classification models performed at the highest level, and were ultimately chosen due to their speed, compared to other high performing models such as SVM or Tensorflow neural nets(NN models using only bits). For the classification model, the best accuracy was achieved after dropping the "intermediate" rated compounds. For regression, all values were included, and the best results were obtained after standardizing the distribution of the target values. Attempts at dimensionality reduction yielded slightly greater results, but were abandoned due to the importance of analyzing residuals and feature importances, as well maintaining robustness for other bacteria searches. Utlimately, due the volume of sparsely populated fragments, variance threshold will be added as pre-processing step in the Streamlit application.

The regression model, after hyperparameter tuning with GridSearchCV, ultimately had an r<sup>2</sup> of 0.764(test), and performed 77.6% greater than a null model. The residuals were used to identify the molecules that the model had the most difficulty predicting efficacy for, and cosine similarity was generated as a tool for researchers to analyze functional groups that need more attention to be properly modeled.

The classfication model, also tuned using GridSearchCV, was ultimately able to predict active vs inactive with an accuracy of 0.98, precision of 0.96, recall equal to 0.97, f1-score of 0.97, and a roc auc of 0.98. Specific "bits" were noted as having significant feature importance, these bits were extracted using the RDKit library, and visualized in notebook 04b. These fragments are important for determing necessary functional groups. The top residuals were also explored, to determine which molecular features the models struggled most with. The fragments were also explored, and the results can be found in notebook 04a.

After successfully performing regression and classification on Acinetobacter baumannii targeting drugs, an app was developed to model the drugs available for any organism, and make a prediction on the Minimum Inhibitory Concentration (MIC) of a new molecule. The app takes in an organism name and a SMILE from the user, and outputs the desired prediction, along with an image of the molecule, and a description of its physical properties. Behind the scenes, the ChEMBL web-client is querying for drugs, and several functions are adding features and modeling. The app was deployed using Streamlit.

Finally, work was put toward developing a unique molecule generator, again using Acinetobacter baumannii classified drugs. The compounds were filtered for effectiveness, and the resulting molecules were fed into an RNN with LSTM layers, using a text generation technique adapted from Deep Learning with Python. This is a work in-progress, but unique, new strings were generated. Furthermore, even when computing on a remote PC with a GPU, only 300 molecules could be used for training, due to the size of the 3d array.

Based on the findings, and to answer the problem statement, a fast, reliable drug discovery pipeline can be, and was, produced. Starting with an organism of interest, a researcher can use the developed pipeline and application to research what drugs have worked, what molecular pieces are important, and make predictions on new compounds.

## Dictionary  
|Feature|Type|Dataset|Description|
|---|---|---|---|  
|**filename**|*object*|Found in bacteria resistance mechanism dataframes.|Represents CARD filename, which can be referenced via command line to access a full .txt genome.|  
|---|---|---|---|  
|**rgi_main.Resistance_Mechanism**|*object*|Found in bacteria resistance mechanism dataframes.|One of six mechanisms a researcher has determined is the primary mechanism by which a bacteria has become resistant to antibiotic treatment. The six values are the following six dictionary items|  
|---|---|---|---|  
|**antibiotic efflux**|*integer: 0*|Found in bacteria resistance mechanism dataframes.|The primary resistance mechanism in Gram-negative bacteria. The development of pump systems that allow bacteria to remove toxic substances, in this case, antibiotics.|   
|---|---|---|---|  
|**antibiotic inactivation**|*integer*: 1|Found in bacteria resistance mechanism dataframes.|The ability of a bacteria to produce an enzyme that deactivates antibiotics.|  
|---|---|---|---|  
|**antibiotic target alteration**|*integer*: 2|Found in bacteria resistance mechanism dataframes.|Also known as target modification. Usually the result of spontaneous mutation, the result of which is the moving of a target area.|   
|---|---|---|---| 
|**antibiotic target replacement**|*integer*: 3|Found in bacteria resistance mechanism dataframes.|Similar to target alteration, but results in a completely different or removed target area.|  
|---|---|---|---| 
|**antibiotic target protection**|*integer*: 4|Found in bacteria resistance mechanism dataframes.|The development of a resistance protein that protects, or "rescues," a bacteria from an antibiotic.|   
|---|---|---|---|  
|**reduced permeability to antibiotic**|*integer*: 5|Found in bacteria resistance mechanism dataframes.|Specific to Gram-negative bacteria. The development of a unique, evolutionarily new, lipid bilayer that prevent antibiotic interaction.|  
|---|---|---|---|  
|**rgi_main.CARD_Protein_Sequence**|*object*|Found in bacteria resistance mechanism dataframes.|A sequence of amino acids representing a protein snippet that has been determined by geneticists to be related to the antibiotic resistance of an organism.|   
|---|---|---|---|  
|**canonical_smiles**|*object*|Found in drug discovery dataframes.|A unique representation of a molecular structure that takes into account all atoms, functional groups, bond pairs, etc.|  
|---|---|---|---|  
|**standard_value**|*float*|Found in drug discovery dataframes.|A standardized value of potency in nM. For antibiotics, this value is the MIC or Minimum Inhibitory Concentration.|   
|---|---|---|---|  
|**bioactivity_binary**|*integer*|Found in drug discovery dataframes.|Active (1) or inactive (0) classification determined by a compounds MIC/standard value. Used as the **target variable** for classification.|  
|---|---|---|---|  
|**bioactivity_multiclass**|*object*|Found in drug discovery dataframes.|Active, intermediate, or inactive classification determined by MIC/standard value, and primarily used for sorting and filtering.|  
|---|---|---|---|  
|**mol_wt**|*float*|Found in drug discovery dataframes.|The weight of a molecule in g/mol.|  
|---|---|---|---|  
|**log_p**|*float*|Found in drug discovery dataframes.|the lipophilicity of a drug, which is the ratio at equilibrium of the concentration of a compound between two phases - oil and a liquid.|  
|---|---|---|---|  
|**proton_donors**|*integer*|Found in drug discovery dataframes.|The number of available hydrogens for bonding.|  
|---|---|---|---|  
|**proton_acceptors**|*integer*|Found in drug discovery dataframes.|The number of available locations for hydrogen bonding.|  
|---|---|---|---|  
|**pMIC**|*integer*|Found in drug discovery dataframes.|A -log10 standardization of the MIC or standard value. Used as the **target variable** for regression.|  
|---|---|---|---|  

## Sources
1. A Deep Learning Approach to Antibiotic Discovery: https://www.cell.com/cell/fulltext/S0092-8674(20)30102-1?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867420301021%3Fshowall%3Dtrue  
2. Antibiotic resistance: bioinformatics-based understanding as a functional strategy for drug design: https://pubs.rsc.org/en/Content/ArticleLanding/2020/RA/D0RA01484B#!divAbstract  
3. MIC database: A collection of antimicrobial compounds from literature: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2823385/  
4. Machine learning-powered antibiotics phenotypic drug discovery: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6428806/  
5. Helping Chemists Discover New Antibiotics:  https://pubs.acs.org/doi/10.1021/acsinfecdis.5b00044  
6. New Statistical Technique for Analyzing MIC-Based Susceptibility Data: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3294928/  
7. Applying machine learning techniques to predict the properties of energetic materials: https://www.nature.com/articles/s41598-018-27344-x  
8. QBMG: quasi-biogenic molecule generator with deep recurrent neural network: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0328-9  
9. Deep Learning with Python by Francois Chollet  
10. Are the physicochemical properties of antibacterial compounds really different from other drugs?: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0143-5  