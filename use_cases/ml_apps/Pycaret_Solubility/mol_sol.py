import streamlit as st
import os
import io
import base64
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem, DataStructs, RDConfig
import pycaret
from pycaret.classification import *

#App layout
st.set_page_config(page_title='ML model comparison using PyCaret', layout='centered')
st.title("Modeling molecular solubility with PyCaret :microscope:")

#Introduction text
st.subheader('This example uses Delaneys Solubility dataset as an example')
st.markdown('The data used in this example is from the paper *ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure* by John Delaney. This paper demonstrates a method to estimate the aqueos solubility of a compound from its molecular structure. The model takes SMILES as an input and calculates features such as logP_octanol, molecular weight, proportion of heavy atoms in aromatic systems, and number of rotatable bonds.')

#Import data
traindir = './solubility.train.sdf'
testdir = './solubility.test.sdf'

#Convert structural data to SMILES
train_mols = [m for m in Chem.SDMolSupplier(traindir)]
test_mols = [m for m in Chem.SDMolSupplier(testdir)]

#Function to generate molecular fingerprints from SMILES
def mol2fp(mol, radi=2, nBits=1024):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radi, nBits=nBits)
    arr = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

#Prediction dictionairy
prop_dict = {
            '(A) low':0,
            '(B) medium':1,
            '(C) high':2
            }

# Preprocessing, splitting, and creatiung dataframes
# Borrrowed from iwatobipen
nBits=1024
columns = [f'fp_{idx}' for idx in range(nBits)] + ['target']
train_x = np.array([mol2fp(m, nBits=nBits) for m in train_mols])
test_x = np.array([mol2fp(m, nBits=nBits) for m in test_mols])
train_target = np.array([prop_dict[m.GetProp('SOL_classification')] for m in train_mols]).reshape(-1,1)
test_target = np.array([prop_dict[m.GetProp('SOL_classification')] for m in test_mols]).reshape(-1,1)
train_data = np.concatenate([train_x, train_target], axis=1)
test_data = np.concatenate([test_x, test_target], axis=1)
train_df = pd.DataFrame(train_data, columns=columns)
test_df = pd.DataFrame(test_data, columns=columns)

# Sidebar intro
st.sidebar.title("Model sandbox")
st.sidebar.markdown("**Choose a machine learning algorithm from below to see details about each**")

# Sidebar model comparison example
if st.button('Start here with the example dataset'):

    # Model comparison intro
    st.header("Model Comparison")
    st.write("After running all the default classifiers included in PyCaret, a dataframe appears below which summerizes the metrics of each classfier.")

    # Initiating model comparison setup and pulling model data
    def model_setup(train_df):
        setup(data=train_df, target='target', silent = True, session_id=123)

    def model_comparison():
        best_model = compare_models()
        results = pull(best_model)
        return results

    # execute example functions
    model_setup(train_df)

    results = model_comparison()

    st.dataframe(results.style.highlight_max(axis=0))

    # Random forest information and results
    st.header("Taking a closer look at the trained model")
    st.write("The random forest classifier was trained using 10-fold stratified cross validation and the metrics for each run as well the mean and standard deviation.")
    rfmodel = create_model('rf')
    rfmodel_results = pull()
    st.write(rfmodel_results)

    # Tuning the model and plotting AUC ROC curves
    st.header("Tuning the random forest classfier")
    st.write("Plotting the ROC curves for the tuned random forest model")
    tuned_rf = tune_model(rfmodel)
    plot_model(tuned_rf, save = True)
    st.image('AUC.png')

    # Feature importance info and plot
    st.subheader("Feature importance plot")
    plot_model(tuned_rf, plot="feature", save = True)
    st.image('Feature Importance.png')

    # Confusion Matrix
    st.header("Confusion Matrix")
    plot_model(tuned_rf, plot="confusion_matrix", save = True)
    st.image('Confusion Matrix.png')

    # Random forest test results
    st.header("The test results are ass but it works")
    st.write("The test results are ass but it works")
    evaluate_model(tuned_rf)
    test = pull()
    st.write(test)

# Logistic regression model
if st.sidebar.button("Logistic regression"):

    st.title("Logistic Regression ⁠— metrics and results")
    st.header("About the Algorithm")
    st.write("The logistic regression algorithm classifies a categorical response (outcome) variable between 1 and 0 based on its relationship with predictive features. In contrast, linear regression outputs response variables that are continuous and can be any real number. Most importantly, linear regression does not output probabilities and instead fits the best hyperplane. Therefore logistic regression is the natural choice for classification problems.")
    st.image("lr.png")
    st.header("Training data model results")

    def model_setup(train_df):
        setup(data=train_df, target='target', silent = True, session_id=123)

    model_setup(train_df)

    lr_model = create_model('lr')
    lr_results = pull()
    st.write(lr_results)

    with st.beta_expander("1. See the AUC ROC curve"):
    #ROC AUC
        st.header("Plotting training AUC ROC curves for logistic regression")
        st.write("Plotting the ROC curves for model")
        plot_model(lr_model, save = True)
        st.image('AUC.png')

    with st.beta_expander("2. See feature importance"):
    # Feature importance
        st.header("Feature importance plot")
        plot_model(lr_model, plot="feature", save = True)
        st.image('Feature Importance.png')

    with st.beta_expander("3. See confusion matrix"):
    # Confusion Matrix
        st.header("Confusion Matrix")
        plot_model(lr_model, plot="confusion_matrix", save = True)
        st.image('Confusion Matrix.png')

    with st.beta_expander("4. See test results"):
    # Test results
        st.header("The test results are ass but it works")
        st.write("The test results are ass but it works")
        evaluate_model(lr_model)
        lr_test = pull()
        st.write(lr_test)

# Light GBM model
if st.sidebar.button("Light GBM"):

    st.title("Light GBM ⁠— metrics and results")
    st.header("About the Algorithm")
    st.write("Light GBM is a gradient boosting framework that uses a tree based learning algorithm. Light GBM grows trees vertically while other algorithms grow trees horizontally meaning that Light GBM grows trees leaf-wise instead of level-wise. Light GBM is prefixed as ‘Light’ because of its high speed as it can train largere datasets while taking less memory to run. LGBM also supports GPU learning Light GBM is sensitive to overfitting and can easily overfit small datasets")
    st.image("lgbm.png")
    st.header("Training data model results")

    def model_setup(train_df):
        setup(data=train_df, target='target', silent = True, session_id=123)

    model_setup(train_df)

    lightgbm_model = create_model('lightgbm')
    lightgbm_results = pull()
    st.write(lightgbm_results)

    with st.beta_expander("1. See the AUC ROC curve"):
    #ROC AUC
        st.header("Plotting training AUC ROC curves for lightgbm")
        st.write("Plotting the ROC curves for model")
        plot_model(lightgbm_model, save = True)
        st.image('AUC.png')

    with st.beta_expander("2. See feature importance"):
    # Feature importance
        st.header("Feature importance plot")
        plot_model(lightgbm_model, plot="feature", save = True)
        st.image('Feature Importance.png')

    with st.beta_expander("3. See confusion matrix"):
    # Confusion Matrix
        st.header("Confusion Matrix")
        plot_model(lightgbm_model, plot="confusion_matrix", save = True)
        st.image('Confusion Matrix.png')

    with st.beta_expander("4. See test results"):
    # Test results
        st.header("The test results are ass but it works")
        st.write("The test results are ass but it works")
        evaluate_model(lightgbm_model)
        lightgbm_test = pull()
        st.write(lightgbm_test)


# Ridge regression model
if st.sidebar.button("Ridge regression"):

    st.title("Ridge regression ⁠— metrics and results")
    st.header("About the Algorithm")
    st.write("Ridge regression can be used to model supervised learning problems and is useful in solving problems where you have a smaller dataset (< 100k samples) or when you have more parameters than samples.")
    st.write("Ridge regression uses L2 regularization which adds the following penalty term to the OLS equation.")
    st.image("ridge.png")
    st.write("The L2 term is equal to the square of the magnitude of the coefficients. In this case if lambda(λ) is zero then the equation is the basic OLS but if it is greater than zero then we add a constraint to the coefficients. This constraint results in minimized coefficients (aka shrinkage) that trend towards zero the larger the value of lambda. Shrinking the coefficients leads to a lower variance and in turn a lower error value. Therefore Ridge regression decreases the complexity of a model but does not reduce the number of variables, it rather just shrinks their effect.")

    st.header("Training data model results")

    def model_setup(train_df):
        setup(data=train_df, target='target', silent = True, session_id=123)

    model_setup(train_df)

    ridge_model = create_model('ridge')
    ridge_results = pull()
    st.write(ridge_results)

    with st.beta_expander("1. See feature importance"):
    # Feature importance
        st.header("Feature importance plot")
        plot_model(ridge_model, plot="feature", save = True)
        st.image('Feature Importance.png')

    with st.beta_expander("2. See confusion matrix"):
    # Confusion Matrix
        st.header("Confusion Matrix")
        plot_model(ridge_model, plot="confusion_matrix", save = True)
        st.image('Confusion Matrix.png')

    with st.beta_expander("3. See test results"):
    # Random forest test results
        st.header("The test results are ass but it works")
        st.write("The test results are ass but it works")
        evaluate_model(ridge_model)
        ridge_test = pull()
        st.write(ridge_test)


# ADA Boost model
if st.sidebar.button("ADA Boost"):

    st.title("ADA Boost ⁠— metrics and results")
    st.header("About the Algorithm")
    st.write("Boosting is a method of converting weak learners into strong learners. In boosting, each new tree is a fit on a modified version of the original data set. The AdaBoost Algorithm begins by training a decision tree in which each observation is assigned an equal weight. After evaluating the first tree, we increase the weights of those observations that are difficult to classify and lower the weights for those that are easy to classify. The second tree is therefore grown on this weighted data. Here, the idea is to improve upon the predictions of the first tree. Our new model is therefore Tree 1 + Tree 2. ")
    st.image("adaboost.jpg")
    st.write("We then compute the classification error from this new 2-tree ensemble model and grow a third tree to predict the revised residuals. We repeat this process for a specified number of iterations. Subsequent trees help us to classify observations that are not well classified by the previous trees. Predictions of the final ensemble model is therefore the weighted sum of the predictions made by the previous tree models.")

    st.header("Training data model results")

    def model_setup(train_df):
        setup(data=train_df, target='target', silent = True, session_id=123)

    model_setup(train_df)

    ada_model = create_model('ada')
    ada_results = pull()
    st.write(ada_results)

    with st.beta_expander("1. See the AUC ROC curve"):
    #ROC AUC
        st.header("Plotting training AUC ROC curves for ADA Boost")
        st.write("Plotting the ROC curves for model")
        plot_model(ada_model, save = True)
        st.image('AUC.png')

    with st.beta_expander("2. See feature importance"):
    # Feature importance
        st.header("Feature importance plot")
        plot_model(ada_model, plot="feature", save = True)
        st.image('Feature Importance.png')

    with st.beta_expander("3. See confusion matrix"):
    # Confusion Matrix
        st.header("Confusion Matrix")
        plot_model(ada_model, plot="confusion_matrix", save = True)
        st.image('Confusion Matrix.png')

    with st.beta_expander("4. See test results"):
    # Test results
        st.header("The test results are ass but it works")
        st.write("The test results are ass but it works")
        evaluate_model(ada_model)
        ada_test = pull()
        st.write(ada_test)


# Gradient Boosting model
if st.sidebar.button("Gradient Boosting"):

    st.title("Gradient Boosting ⁠— metrics and results")
    st.header("About the Algorithm")
    st.write("Gradient Boosting is similar to AdaBoost in that they both use an ensemble of decision trees to predict a target label. However, unlike AdaBoost, the Gradient Boost trees have a depth larger than 1. In practice, you’ll typically see Gradient Boost being used with a maximum number of leaves of between 8 and 32.")
    st.image("gbc.png")
    st.write("Gradient Boosting trains many models in a gradual, additive and sequential manner. The major difference between AdaBoost and Gradient Boosting Algorithm is how the two algorithms identify the shortcomings of weak learners (eg. decision trees). While the AdaBoost model identifies the shortcomings by using high weight data points, gradient boosting performs the same by using gradients in the loss function (y=ax+b+e , e needs a special mention as it is the error term). The loss function is a measure indicating how good are model’s coefficients are at fitting the underlying data.")

    st.header("Training data model results")

    def model_setup(train_df):
        setup(data=train_df, target='target', silent = True, session_id=123)

    model_setup(train_df)

    gbc_model = create_model('gbc')
    gbc_results = pull()
    st.write(gbc_results)

    with st.beta_expander("1. See the AUC ROC curve"):
    #ROC AUC
        st.header("Plotting training AUC ROC curves for Gradient Boosting")
        st.write("Plotting the ROC curves for model")
        plot_model(gbc_model, save = True)
        st.image('AUC.png')

    with st.beta_expander("2. See feature importance"):
    # Feature importance
        st.header("Feature importance plot")
        plot_model(gbc_model, plot="feature", save = True)
        st.image('Feature Importance.png')

    with st.beta_expander("3. See confusion matrix"):
    # Confusion Matrix
        st.header("Confusion Matrix")
        plot_model(gbc_model, plot="confusion_matrix", save = True)
        st.image('Confusion Matrix.png')

    with st.beta_expander("4. See test results"):
    # Test results
        st.header("The test results are ass but it works")
        st.write("The test results are ass but it works")
        evaluate_model(gbc_model)
        gbc_test = pull()
        st.write(gbc_test)


# K Nearest Neighbors model
if st.sidebar.button("K Nearest Neighbors"):

    st.title("K Nearest Neighbors ⁠— metrics and results")
    st.header("About the Algorithm")
    st.header("Training data model results")

    def model_setup(train_df):
        setup(data=train_df, target='target', silent = True, session_id=123)

    model_setup(train_df)

    knn_model = create_model('knn')
    knn_results = pull()
    st.write(knn_results)

    with st.beta_expander("1. See the AUC ROC curve"):
    #ROC AUC
        st.header("Plotting training AUC ROC curves for K Nearest Neighbors")
        st.write("Plotting the ROC curves for model")
        st.write("The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.")
        st.write("Notice in the image above that most of the time, similar data points are close to each other. The KNN algorithm hinges on this assumption being true enough for the algorithm to be useful. KNN captures the idea of similarity (sometimes called distance, proximity, or closeness) with some mathematics we might have learned in our childhood— calculating the distance between points on a graph. There are other ways of calculating distance, and one way might be preferable depending on the problem we are solving. However, the straight-line distance (also called the Euclidean distance) is a popular and familiar choice.")
        st.image("knn.png")
        plot_model(knn_model, save = True)
        st.image('AUC.png')

    with st.beta_expander("2. See confusion matrix"):
    # Confusion Matrix
        st.header("Confusion Matrix")
        plot_model(knn_model, plot="confusion_matrix", save = True)
        st.image('Confusion Matrix.png')

    with st.beta_expander("3. See test results"):
    #  test results
        st.header("The test results are ass but it works")
        st.write("The test results are ass but it works")
        evaluate_model(knn_model)
        knn_test = pull()
        st.write(knn_test)


# SVM - Linear Kernel model
if st.sidebar.button("SVM ⁠— Linear Kernel"):

    st.title("SVM Linear Kernel ⁠— metrics and results")
    st.header("About the Algorithm")
    st.write("A simple linear SVM classifier works by making a straight line between two classes. That means all of the data points on one side of the line will represent a category and the data points on the other side of the line will be put into a different category. SVM chooses the decision boundary that maximizes the distance from the nearest data points of all the classes. The decision boundary created by SVMs is called the maximum margin classifier or the maximum margin hyperplane.")
    st.image("svm.png")
    st.write("It's a great option when you are working with smaller datasets that have tens to hundreds of thousands of features. It will typically find more accurate results when compared to other algorithms because of their ability to handle small, complex datasets.")
    st.header("Training data model results")

    def model_setup(train_df):
        setup(data=train_df, target='target', silent = True, session_id=123)

    model_setup(train_df)

    svm_model = create_model('svm')
    svm_results = pull()
    st.write(svm_results)

    with st.beta_expander("1. See feature importance"):
    # Feature importance
        st.header("Feature importance plot")
        plot_model(svm_model, plot="feature", save = True)
        st.image('Feature Importance.png')

    with st.beta_expander("2. See confusion matrix"):
    # Confusion Matrix
        st.header("Confusion Matrix")
        plot_model(svm_model, plot="confusion_matrix", save = True)
        st.image('Confusion Matrix.png')

    with st.beta_expander("3. See test results"):
    # Test results
        st.header("The test results are ass but it works")
        st.write("The test results are ass but it works")
        evaluate_model(svm_model)
        svm_test = pull()
        st.write(svm_test)


# Decision Tree model
if st.sidebar.button("Decision Tree"):

    st.title("Decision Tree ⁠— metrics and results")
    st.header("About the Algorithm")
    st.write("The Decision Tree algorithm is a supervised learning algorithm that can be used for solving regression and classification problems. The algorithm creates a training model to predict the class or value of the target variable by learning simple decision rules inferred from the training data")
    st.image("dt.png")
    st.write("Decision trees classify the examples by sorting them down the tree from the root to some leaf/terminal node, with the leaf/terminal node providing the classification of the example. Each node in the tree acts as a test case for some attribute, and each edge descending from the node corresponds to the possible answers to the test case.")
    st.header("Training data model results")

    def model_setup(train_df):
        setup(data=train_df, target='target', silent = True, session_id=123)

    model_setup(train_df)

    dt_model = create_model('lr')
    dt_results = pull()
    st.write(dt_results)

    with st.beta_expander("1. See the AUC ROC curve"):
    #ROC AUC
        st.header("Plotting training AUC ROC curves for Decision Tree")
        st.write("Plotting the ROC curves for model")
        plot_model(dt_model, save = True)
        st.image('AUC.png')

    with st.beta_expander("2. See feature importance"):
    # Feature importance
        st.header("Feature importance plot")
        plot_model(dt_model, plot="feature", save = True)
        st.image('Feature Importance.png')

    with st.beta_expander("3. See confusion matrix"):
    # Confusion Matrix
        st.header("Confusion Matrix")
        plot_model(dt_model, plot="confusion_matrix", save = True)
        st.image('Confusion Matrix.png')

    with st.beta_expander("4. See test results"):
    # Test results
        st.header("The test results are ass but it works")
        st.write("The test results are ass but it works")
        evaluate_model(dt_model)
        dt_test = pull()
        st.write(dt_test)

# Naive Bayes model
if st.sidebar.button("Naive Bayes"):

    st.title("Naive Bayes ⁠— metrics and results")
    st.header("About the Algorithm")
    st.header("Training data model results")

    def model_setup(train_df):
        setup(data=train_df, target='target', silent = True, session_id=123)

    model_setup(train_df)

    nb_model = create_model('nb')
    nb_results = pull()
    st.write(nb_results)

    with st.beta_expander("1. See the AUC ROC curve"):
    #ROC AUC
        st.header("Plotting training AUC ROC curves for Decision Tree")
        st.write("Plotting the ROC curves for model")
        plot_model(nb_model, save = True)
        st.image('AUC.png')

    with st.beta_expander("2. See confusion matrix"):
    # Confusion Matrix
        st.header("Confusion Matrix")
        plot_model(nb_model, plot="confusion_matrix", save = True)
        st.image('Confusion Matrix.png')

    with st.beta_expander("3. See test results"):
    # Test results
        st.header("The test results are ass but it works")
        st.write("The test results are ass but it works")
        evaluate_model(nb_model)
        nb_test = pull()
        st.write(nb_test)
