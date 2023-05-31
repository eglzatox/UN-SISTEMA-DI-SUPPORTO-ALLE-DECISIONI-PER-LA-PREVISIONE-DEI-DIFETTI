#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 19:17:34 2023

@author: marcomulas
"""

#%%

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# Sklearn modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

# Other modules
import re
#%%

# **Reading the data**
df = pd.read_csv("ORIGINAL_defect_dataset.csv")

# Return a random sample of items from an axis of object
df.sample(10)
#%%

# **Data Exploration**

# Returns a tuple (number of rows, number of columns)representing the dimensionality of the DataFrame
df.shape

# Returns the number of observations
df.size

# Is used to get a concise summary of the DataFrame including the index dtype and columns, non-null values and memory usage
df.info()

# This function returns the first n rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it.
df.head()

# Display column names in the dataset
df.columns

# Generate descriptive statistics.
df.describe()

# Create arrays with descriptions so we can see all in the variable ecplorer
describe = df.describe()

# Returns description of the data in the DataFrame, now including the object
df.describe(include="object")
describe_object = df.describe(include="object")
#%%

# **Data Cleaning**

# Deleting the row with index 327
df = df.drop(327)

# DataFrame indexes reset
df = df.reset_index(drop=True)

# Delete irrelevant features
df.drop(['ID','SKU','ANNO','Product Division','New Zone','RTW F/C','Case id','Case Header number','Tech. Classification type',
         'Defect Description','CN Management','Size','Cites','MTO','Product: Material Type Description','Purchase Store',
         'Arrival Date','FEEDBACK PRODUCTION',' DEFECT CLUSTER FJ','Note','DEF','Out Sesto','Giorni','ESITO','NOTE BAR',
         'Probabilità','Tipo Materiale', 'Probabilità Rischio per Modello Parte','Gravità Difettosità',
         'Matching Probabilità Modello Parte + Gravità Difetto'], axis=1, inplace=True)

# Features' arrays
feature_list = list(df.columns)

# Let's see all the labels that are object
df.select_dtypes('O').info()

# Function to get count of missing values in each column
missing_values_df = df.isna().sum()
missing_values_df

# Count the total number of missing values
print(df.isnull().values.sum())  # there are  504 nan


missing_values_df = df[['RTW F/C','Cites','Purchase Store','Arrival Date',' DEFECT CLUSTER FJ','Note','DEF', 'Out Sesto','Giorni','ESITO','NOTE BAR']].isnull().sum()
missing_values_df

# Plot of percentage of missing values
plt.xticks(rotation='90')
sns.barplot(x=missing_values_df.index, y=missing_values_df)
plt.title('Percentage of Missing Values')
#%%

# Analysis of all the variables

# MESE
mese = df['MESE'].value_counts(dropna=False)
#%%

# Stock Origin
stock_origin = df['Stock Origin'].value_counts(dropna=False)
#%%

# Replaces nan values with "Not defined" 
df['ESITO FLUSSO'] = df['ESITO FLUSSO'].fillna('Not defined')

# ESITO FLUSSO
esito_flusso = df['ESITO FLUSSO'].value_counts(dropna=False)
#%%

# Zone
zone = df['Zone'].value_counts(dropna=False)
#%%

# Market
market = df['Market'].value_counts(dropna=False)
#%%

# Defines the regular expression to search for numbers that are not dates
regex = r'^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$'

# Replaces numbers that are not given with NaN
df['Created Date'] = df['Created Date'].apply(lambda x: np.nan if re.match(regex, str(x)) is None else x)

# Replaces nan values with "Not defined" 
df['Created Date'] = df['Created Date'].fillna('Not defined')

# Created Date
created_date = df['Created Date'].value_counts(dropna=False)
#%%

# Actual Status
actual_status = df['Actual Status'].value_counts(dropna=False)
#%%

# Case Header Type
case_header_type = df['Case Header type'].value_counts(dropna=False)
#%%

# Replaces nan values with "Not defined" 
df['Store Reference: Store Name'] = df['Store Reference: Store Name'].fillna('Not defined')

# Store Reference: Store Name
store_name = df['Store Reference: Store Name'].value_counts(dropna=False)
#%%

# Classification defect main location
class_defect_main_location = df['Classification defect main location'].value_counts(dropna=False)
#%%

# Replaces nan values with "Not defined" 
df['Classification defect detailed location'] = df['Classification defect detailed location'].fillna('Not defined')

# Classification defect detailed location
class_defect_detailed_location = df['Classification defect detailed location'].value_counts(dropna=False)
#%%

# Substitution of the number 0 with 'Not defined'
df['Classification defect type'] = df['Classification defect type'].replace({'0': 'Not defined'}, regex=True)

# Classification defect type
class_defect_type = df['Classification defect type'].value_counts(dropna=False)
#%%

# Collection
collection = df['Collection'].value_counts(dropna=False)
#%%

# Product Category
product_category= df['Product Category'].value_counts(dropna=False)
#%%

# Removal of the X
df['Product Line'] = df['Product Line'].str.replace('X', '')

# Product Line
product_line = df['Product Line'].value_counts(dropna=False)
#%%

# Product Type
product_type = df['Product Type'].value_counts(dropna=False)
#%%

# Defines the regular expression to search for numbers that are not dates
regex = r'^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$'

# Replaces numbers that are not given with NaN
df['Purchase Date'] = df['Purchase Date'].apply(lambda x: np.nan if re.match(regex, str(x)) is None else x)

# Replaces nan values with "Not defined" 
df['Purchase Date'] = df['Purchase Date'].fillna('Not defined')

# Purchase Date
purchase_date = df['Purchase Date'].value_counts(dropna=False)
#%%

# Convert values to strings
df['Days from Purchase to Complaints '] = df['Days from Purchase to Complaints '].astype(str)

# Replaces nan values with "Not defined" 
df['Days from Purchase to Complaints '] = df['Days from Purchase to Complaints '].fillna('Not defined')

# Days from Purchase to Complaints 
days = df['Days from Purchase to Complaints '].value_counts(dropna=False)
#%%

# Lifespan
lifespan = df['Lifespan'].value_counts(dropna=False)
#%%

# Retail status
retail_status = df['Retail status'].value_counts(dropna=False)
#%%

# Substitution of the letter X with the letter F 
df['Deplation status'] = df['Deplation status'].replace({'X': 'f'}, regex=True)

# Deplation status
deplation_status = df['Deplation status'].value_counts(dropna=False)
#%%

# Replaces nan values with "Not defined" 
df['Launch Season in Retail'] = df['Launch Season in Retail'].fillna('Not defined')

# Launch Season in Retail
launch_season_retail = df['Launch Season in Retail'].value_counts(dropna=False)
#%%

# Defines the regular expression to search for numbers that are not dates
regex = r'^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$'

# Replaces numbers that are not given with NaN
df['Launch Date'] = df['Launch Date'].apply(lambda x: np.nan if re.match(regex, str(x)) is None else x)

# Replace the value "1/1/2100" with NaN
df['Launch Date'] = df['Launch Date'].replace('1/1/2100', pd.NA)

# Replaces nan values with "Not defined" 
df['Launch Date'] = df['Launch Date'].fillna('Not defined')

# Launch Date
launch_date = df['Launch Date'].value_counts(dropna=False)
#%%

# Substitution of the letter X with the letter F 
df['Note on Defect'] = df['Note on Defect'].replace({'X': 'f'}, regex=True)

# Replaces nan values with "Not defined" 
df['Note on Defect'] = df['Note on Defect'].fillna('Not defined')

# Note on Defect
note_on_efect = df['Note on Defect'].value_counts(dropna=False)
#%%
# Substitution of the letter GraXXiata with Graffiata
df['Classificazione BAR'] = df['Classificazione BAR'].replace({'GraXXiata': 'Graffiata'}, regex=True)

# Substitution of the letter X 
df['Classificazione BAR'] = df['Classificazione BAR'].replace({'X': ''}, regex=True)

# Replaces nan values with "Not defined" 
df['Classificazione BAR'] = df['Classificazione BAR'].fillna('Not defined')

# Classificazione BAR
classificazione_BAR = df['Classificazione BAR'].value_counts(dropna=False)
#%%

# Descrizione parte
descrizione_parte = df['Descrizione parte'].value_counts(dropna=False)
#%%

# Material Type
Material_Type = df['Material Type'].value_counts(dropna=False)
#%%

# Matching
Matching = df['Matching'].value_counts(dropna=False)
#%%

# Check the number of observations after the data cleaning
df.size

# Check whether it has eliminated all the missing values
missing_values_df = df.isna().sum()
missing_values_df

# Boxplot for numerical features to see if there are outliers
fig = plt.figure(figsize=(10, 7))
boxplot = df.boxplot(column=['MESE'],grid=False, fontsize=15)
boxplot2 = df.boxplot(column=['Matching'],grid=False, fontsize=15)
#%%

# Encoding categorical variables
# Select only categorical features
features_categoriche = df.select_dtypes(include="object")

# Stock Origin
Stock_origin_la= LabelEncoder()
df['Stock Origin'] = Stock_origin_la.fit_transform(df['Stock Origin'])
Stock_origin_la.classes_
df['Stock Origin'].value_counts(dropna=False)

# ESITO FLUSSO
ESITO_FLUSSO_la= LabelEncoder()
df['ESITO FLUSSO'] = ESITO_FLUSSO_la.fit_transform(df['ESITO FLUSSO'])
ESITO_FLUSSO_la.classes_
df['ESITO FLUSSO'].value_counts(dropna=False)

# Zone
Zone_la= LabelEncoder()
df['Zone'] = Zone_la.fit_transform(df['Zone'])
Zone_la.classes_
df['Zone'].value_counts(dropna=False)

# Market
Market_la= LabelEncoder()
df['Market'] = Market_la.fit_transform(df['Market'])
Market_la.classes_
df['Market'].value_counts(dropna=False)

# Created Date   
Created_Date_la= LabelEncoder()
df['Created Date'] = Created_Date_la.fit_transform(df['Created Date'])
Created_Date_la.classes_
df['Created Date'].value_counts(dropna=False)

# Actual Status
Actual_Status_la= LabelEncoder()
df['Actual Status'] = Actual_Status_la.fit_transform(df['Actual Status'])
Actual_Status_la.classes_
df['Actual Status'].value_counts(dropna=False)

# Case Header Type      
Case_Header_type_la= LabelEncoder()
df['Case Header type'] = Case_Header_type_la.fit_transform(df['Case Header type'])
Case_Header_type_la.classes_
df['Case Header type'].value_counts(dropna=False)

# Store Reference: Store Name
Store_Reference_la= LabelEncoder()
df['Store Reference: Store Name'] = Store_Reference_la.fit_transform(df['Store Reference: Store Name'])
Store_Reference_la.classes_
df['Store Reference: Store Name'].value_counts(dropna=False)

# Classification defect main location 
Class_defect_main_la= LabelEncoder()
df['Classification defect main location'] = Class_defect_main_la.fit_transform(df['Classification defect main location'])
Class_defect_main_la.classes_
df['Classification defect main location'].value_counts(dropna=False)

# Classification defect detailed location
Class_defect_detailed_la= LabelEncoder()
df['Classification defect detailed location'] = Class_defect_detailed_la.fit_transform(df['Classification defect detailed location'])
Class_defect_detailed_la.classes_
df['Classification defect detailed location'].value_counts(dropna=False)

# Classification defect type 
Class_defect_type_la= LabelEncoder()
df['Classification defect type'] = Class_defect_type_la.fit_transform(df['Classification defect type'])
Class_defect_type_la.classes_
df['Classification defect type'].value_counts(dropna=False)

# Collection
Collection_la= LabelEncoder()
df['Collection'] = Collection_la.fit_transform(df['Collection'])
Collection_la.classes_
df['Collection'].value_counts(dropna=False)

# Product Category
Product_Category_la= LabelEncoder()
df['Product Category'] =  Product_Category_la.fit_transform(df['Product Category'])
Product_Category_la.classes_
df['Product Category'].value_counts(dropna=False)

# Product Line 
Product_Line_la= LabelEncoder()
df['Product Line'] = Product_Line_la.fit_transform(df['Product Line'])
Product_Line_la.classes_
df['Product Line'].value_counts(dropna=False)

# Product Type
Product_Type_la= LabelEncoder()
df['Product Type'] = Product_Type_la.fit_transform(df['Product Type'])
Product_Type_la.classes_
df['Product Type'].value_counts(dropna=False)

# Purchase Date
Purchase_Date_la= LabelEncoder()
df['Purchase Date'] = Purchase_Date_la.fit_transform(df['Purchase Date'])
Purchase_Date_la.classes_
df['Purchase Date'].value_counts(dropna=False)

# Days from Purchase to Complaints 
days_la= LabelEncoder()
df['Days from Purchase to Complaints '] = days_la.fit_transform(df['Days from Purchase to Complaints '])
days_la.classes_
df['Days from Purchase to Complaints '].value_counts(dropna=False)

# Lifespan
Lifespan_la= LabelEncoder()
df['Lifespan'] = Lifespan_la.fit_transform(df['Lifespan'])
Lifespan_la.classes_
df['Lifespan'].value_counts(dropna=False)

# Retail status
Retail_status_la= LabelEncoder()
df['Retail status'] = Retail_status_la.fit_transform(df['Retail status'])
Retail_status_la.classes_
df['Retail status'].value_counts(dropna=False)

# Deplation status 
Deplation_status_la= LabelEncoder()
df['Deplation status'] = Deplation_status_la.fit_transform(df['Deplation status'])
Deplation_status_la.classes_
df['Deplation status'].value_counts(dropna=False)

# Launch Season in Retail
Launch_Season_Retail_la= LabelEncoder()
df['Launch Season in Retail'] = Launch_Season_Retail_la.fit_transform(df['Launch Season in Retail'])
Launch_Season_Retail_la.classes_
df['Launch Season in Retail'].value_counts(dropna=False)

# Launch Date     
Launch_Date_la= LabelEncoder()
df['Launch Date'] = Launch_Date_la.fit_transform(df['Launch Date'])
Launch_Date_la.classes_
df['Launch Date'].value_counts(dropna=False)

# Note on Defect
Note_Defect_la= LabelEncoder()
df['Note on Defect'] = Note_Defect_la.fit_transform(df['Note on Defect'])
Note_Defect_la.classes_
df['Note on Defect'].value_counts(dropna=False)

# Classificazione BAR    
class_BAR_la= LabelEncoder()
df['Classificazione BAR'] = class_BAR_la.fit_transform(df['Classificazione BAR'])
class_BAR_la.classes_
df['Classificazione BAR'].value_counts(dropna=False)

# Descrizione parte
Descrizione_parte_la= LabelEncoder()
df['Descrizione parte'] = Descrizione_parte_la.fit_transform(df['Descrizione parte'])
Descrizione_parte_la.classes_
df['Descrizione parte'].value_counts(dropna=False)

# Material Type
Material_Type_la= LabelEncoder()
df['Material Type'] = Descrizione_parte_la.fit_transform(df['Material Type'])
Descrizione_parte_la.classes_
df['Material Type'].value_counts(dropna=False)

# Check if there are still object variables
df.info()
#%%

# Extraction of the y
y = df['Matching']

# Drop the y of our dataset
df = df.drop('Matching', axis=1)

# Check if y has missing values
y.isna().sum()  # no
#%%

# **Statistical Analysis**

# Check if I have an unbalanced dataset
# Replace values with labels
y_labels = y.replace({6: 'ALTO', 5: 'ALTO', 4: 'MEDIO', 3: 'MEDIO', 1: 'BASSO', 2: 'BASSO'})

# Save the results obtained by the value_counts() function
z = y_labels.value_counts()

# Specify labels on the x-axis of the graph
xdata3 = ['ALTO','MEDIO','BASSO']

# Create the plot
fig2 = plt.figure(figsize=(10, 7))
plt.bar(xdata3, z)
plt.show()
# Is unbalanced in favor of the ALTO class, penalizing MEDIO and especially the BASSO class


# Correlation of problematic variables with Matching 
corr = df.corr().round(2)
# Correlation between features and Matching 
df['MESE'].corr(y)
df['Stock Origin'].corr(y)
df['ESITO FLUSSO'].corr(y)
df['Zone'].corr(y)
df['Market'].corr(y)
df['Created Date'].corr(y)
df['Actual Status'].corr(y)
df['Case Header type'].corr(y)
df['Store Reference: Store Name'].corr(y)
df['Classification defect main location'].corr(y)
df['Classification defect type'].corr(y)
df['Collection'].corr(y)
df['Product Category'].corr(y)
df['Product Line'].corr(y)
df['Product Type'].corr(y)
df['Purchase Date'].corr(y)
df['Days from Purchase to Complaints '].corr(y)
df['Lifespan'].corr(y)
df['Retail status'].corr(y)
df['Deplation status'].corr(y)
df['Launch Season in Retail'].corr(y)
df['Launch Date'].corr(y)
df['Note on Defect'].corr(y)
df['Classificazione BAR'].corr(y)
df['Descrizione parte'].corr(y)
df['Material Type'].corr(y)

correlations = df.corr()['Matching'].sort_values(ascending=False)
print(correlations)

# Correlation between features of the dataset
rounded_corr_matrix = df.corr().round(2)

# Heatmap on all variables
plt.figure(figsize=(20,20))
heatmap = sns.heatmap(rounded_corr_matrix, annot=True, cmap="YlGnBu", annot_kws={"fontsize": 12})
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 20}, pad=20)
plt.show()

# Choose few features,the ones that are most correlated
features = ['Product Line','Product Type','Classificazione BAR','Descrizione parte']
# Subset the correlation matrix to include only the selected features
subset = df[features].corr()
# Plot the heatmap for the subset of features
heatmap = sns.heatmap(subset, annot=True)
heatmap.set_title('Correlation Heatmap for Selected Features', fontdict={'fontsize': 15}, pad=12)
#%%

# Feature selection

# Define x
x = df

# Mutual information selection
# Selecting the 4 features with the highest value of mutual information
mic = SelectKBest(score_func=mutual_info_classif,k=4)

# Adapt the function switch to the features and output provided
mic.fit(x,y)

# Created a pandas Series object with the feature scores obtained from the selector
feature_MI_score = pd.Series(mic.scores_,index=x.columns)

# Sorted in descending order to show the best features 
sorted_feature_MI_score = feature_MI_score.sort_values(ascending=False)

# Select the top 4 features
top_features = feature_MI_score.sort_values(ascending=False)[:4]
print(top_features)

# Plot the mutual information score for the top features
plt.figure(figsize=(11, 6))
plt.barh(top_features.index, top_features.values)
plt.title('Top 4 Features with Mutual Information Score')
plt.xlabel('Mutual Information Score')
plt.ylabel('Features')
plt.show()

# Selection of the best 4 variables through the process of mutual information 
X_selected = mic.transform(x)
#%%

# Splitting the dataset into training and test set (K-Fold cross validation)
kf = KFold(n_splits=5)
kf

for train_index, test_index in kf.split(X_selected):
    X_train, X_test = X_selected[train_index], X_selected[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(train_index, test_index)
#%%
# KNN

# Check which k as the most accuracy for KNN
# Choose k between 1 to 30
k_range = range(1, 30)
k_scores = []

# Use iteration to calculate different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_selected, y, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
    
# Plot to view the results
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Classifier - Cross-Validated Accuracy')
plt.grid(False)
plt.gca().set_facecolor('white')
plt.show()
# As we can see the bestis K=1


# Use the same model as before
knn = KNeighborsClassifier(n_neighbors = 1)

# X_selected and y will automatically devided by 5 folder, the scoring I will still use the accuracy
scores_knn = cross_val_score(knn, X_selected, y, cv=kf, scoring='accuracy')

# Print all 5 times scores 
print(scores_knn)  # [0.82758621 0.60465116 0.6627907  0.68604651 0.59302326]

# Average about these five scores to get more accuracy score
print(scores_knn.mean()) # 0.6748195669607057



# Prediction on the test set
y_pred = cross_val_predict(knn, X_selected, y, cv=kf)

# Classification report
print(classification_report(y, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()























# Convert labels to a binary representation
y_bin = label_binarize(y, classes=np.unique(y))

# Calculate ROC curves for each class
n_classes = y_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], knn.fit(X_selected, y).predict_proba(X_selected)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calculate the average ROC curve
fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), knn.fit(X_selected, y).predict_proba(X_selected).ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curve for each class
plt.figure()
lw = 2
colors = ['pink', 'black', 'cornflowerblue', 'red', 'green', 'blue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])

# Plot the micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='Micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))

# Plot the chance line
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
#%%

# SVM

# Define the range of values ​​for parameter C
C_range = [0.01, 0.1, 1, 10, 100]

# List for saving average accuracies
C_scores = []

# Loop on the values ​​of C
for C_value in C_range:
    clf = svm.SVC(kernel='linear', C=C_value, probability=True)
    scores = cross_val_score(clf, X_selected, y, cv=kf, scoring='accuracy')
    C_scores.append(scores.mean())

# Plot to view the results
plt.plot(C_range, C_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Value of C')
plt.ylabel('Cross-Validated Accuracy')
plt.title('SVM Classifier - Cross-Validated Accuracy')
plt.grid(False)
plt.show()
# As we can see the best is C=1

# Use the same model as before
clf = svm.SVC(kernel='linear', C=1, probability=True)

# X_selected and y will automatically devided by 5 folder, the scoring I will still use the accuracy
scores_clf = cross_val_score(clf, X_selected, y, cv=kf, scoring='accuracy')

# Print all 5 times scores 
print(scores_clf)  # [0.42528736 0.27906977 0.39534884 0.58139535 0.06976744]

# Average about these five scores to get more accuracy score
print(scores_clf.mean()) # 0.3501737503341352



# Prediction on the test set
y_pred = cross_val_predict(clf, X_selected, y, cv=kf)

# Classification report
print(classification_report(y, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Convert labels to a binary representation
y_bin = label_binarize(y, classes=np.unique(y))

# Calculate ROC curves for each class
n_classes = y_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], knn.fit(X_selected, y).predict_proba(X_selected)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calculate the average ROC curve
fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), clf.fit(X_selected, y).predict_proba(X_selected).ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curve for each class
plt.figure()
lw = 2
colors = ['pink', 'black', 'cornflowerblue', 'red', 'green', 'blue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])

# Plot the micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='Micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))

# Plot the chance line
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
#%%

# RANDOM FOREST

# Check which estimators as the most accuracy for RANDOM FOREST
rf_range = [50, 100, 150, 200, 250, 300, 350]
rf_scores = []

for e in rf_range:
    rf = RandomForestClassifier(n_estimators=e)
    scores = cross_val_score(rf, X_selected, y, cv=kf, scoring='accuracy')
    rf_scores.append(scores.mean())

# Plot to view the results
plt.plot(rf_range, rf_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Value of estimators')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Random Forest Classifier - Cross-Validated Accuracy')
plt.grid(False)
plt.show()
# Many plot are given, the value tha gives higher accuracy is different from plot to plot

    
# Use the same model as before
rf= RandomForestClassifier(n_estimators=100)

# X_selected and y will automatically devided by 5 folder, the scoring I will still use the accuracy
scores_rf = cross_val_score(rf, X_selected, y, cv=kf, scoring='accuracy')

# Print all 5 times scores 
print(scores_rf)  # [0.87356322 0.87209302 0.72093023 0.70930233 0.61627907]

# Average about these five scores to get more accuracy score
print(scores_rf.mean()) # 0.7584335739107191


# Prediction on the test set
y_pred_rf = cross_val_predict(rf, X_selected, y, cv=kf)

# Classification report
print(classification_report(y, y_pred_rf))

# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred_rf)
print(conf_matrix)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Convert labels to a binary representation
y_bin = label_binarize(y, classes=np.unique(y))

# Calculate ROC curves for each class
n_classes = y_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], rf.fit(X_selected, y).predict_proba(X_selected)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calculate the average ROC curve
fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), rf.fit(X_selected, y).predict_proba(X_selected).ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curve for each class
plt.figure()
lw = 2
colors = ['pink', 'black', 'cornflowerblue', 'red', 'green', 'blue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])

# Plot the micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='Micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))

# Plot the chance line
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
#%%

# DECISION TREE

# Check which maxdepth as the most accuracy for DECISION TREE
dt_range = [1,2,3,4,5,6,7,8,9,10]
dt_scores = []

# Use iteration to calculate different maxdepth in models, then return the average accuracy based on the cross validation
for maxdepth in dt_range:
    dt = DecisionTreeClassifier(max_depth=maxdepth)
    scores = cross_val_score(dt, X_selected, y, cv=kf, scoring='accuracy')
    dt_scores.append(scores.mean())
    
# Plot to view the results
plt.plot(dt_range, dt_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Value of MaxDepth')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Decision Tree Classifier - Cross-Validated Accuracy')
plt.grid(False)
plt.show()
# As we can see the best Maxdepth is 10

# Use the same model as before
dt = DecisionTreeClassifier(max_depth=(10))

# X_selected and y will automatically devided by 5 folder, the scoring I will still use the accuracy
scores_dt = cross_val_score(dt, X_selected, y, cv=kf, scoring='accuracy')

# Print all 5 times scores 
print(scores_dt)  # [0.89655172 0.84883721 0.74418605 0.68604651 0.61627907]

# Average about these five scores to get more accuracy score
print(scores_dt.mean()) # 0.7583801122694466


# Prediction on the test set
y_pred_dt = cross_val_predict(rf, X_selected, y, cv=kf)

# Classification report
print(classification_report(y, y_pred_dt))

# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred_dt)
print(conf_matrix)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Convert labels to a binary representation
y_bin = label_binarize(y, classes=np.unique(y))

# Calculate ROC curves for each class
n_classes = y_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], dt.fit(X_selected, y).predict_proba(X_selected)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calculate the average ROC curve
fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), dt.fit(X_selected, y).predict_proba(X_selected).ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curve for each class
plt.figure()
lw = 2
colors = ['pink', 'black', 'cornflowerblue', 'red', 'green', 'blue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])

# Plot the micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='Micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))

# Plot the chance line
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()













































