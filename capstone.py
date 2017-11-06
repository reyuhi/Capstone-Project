<<<<<<< HEAD
# Author: Renan Hidani
# Udacity Capstone Machine Learning Nanodegree
# Porto Seguro's Safe Driver Prediction Competition

# Import Numpy and Pandas Library
import numpy as np
import pandas as pd

# Import Seaborn
import seaborn as sns
#%matplotlib inline

# Pyplot
#import matplotlib as mpl
import matplotlib.pyplot as plt

# Import Metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Import Library
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier,PassiveAggressiveClassifier

# Ignore Warning
import warnings
warnings.filterwarnings("ignore")

# Features available inside train and test files
features_train = ['id','target','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']
features_test = ['id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']

# Import Data, identify the missing values as nan
train_data = pd.read_csv('train.csv',na_values = -1)
test_data = pd.read_csv('test.csv', na_values = -1)

# Get the ID Column for train and test data
train_id = train_data[['id']]
test_id = test_data[['id']]

# Get the target Column for train data
train_target = train_data[['target']]

# Bar Graph
sns.countplot( x='target', data=train_data, palette="Greens_d")

# History Plot
train_data.hist(grid = False,figsize=(20, 20))

#Scatterplot Matrix
#sns.set(style="ticks")
#sns.pairplot(train_data, hue="species")

# Factor Plot
#sns.factorplot(x="target", y="ps_car_13", data=train_data)

#Check Data Type
train_data.dtypes

# Check for Missing Values
train_data.isnull().any()

# Count Amount of missing values in each feature
train_data.isnull().sum()

# Visualize Missing Data
# Import missingno library
import missingno as msno

msno.matrix(train_data, figsize = (15,10),fontsize = 9, labels = True)

# Correlation Plot
#plt.matshow(train_data.corr()) - Not good for visualization
sns.set(style="white")
# Compute the correlation matrix
corr = train_data.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Dendrogram
msno.dendrogram(train_data)

# Remove Nan Features (-1) in Rows
train_data_clean = train_data.dropna()
test_data_clean = test_data.dropna()

# Amount of Rows lost after Removing Nan Features in Rows
len(train_data.index)
len(train_data_clean.index)

#"print Amount of Rows without missing values:" "{0:.1f}%".format(len(train_data_clean.index)*100/len(train_data.index))

# Test algorithms with Nan Values
val_nan = True

# Import SMOTE to balance data
from imblearn.over_sampling import SMOTE

# Import Train Test Split Library
from sklearn.model_selection import train_test_split

train_data_sm = train_data_clean.drop(['id','target'],axis =1)
train_target_sm = train_data_clean[['target']]

x_train, x_val, y_train, y_val = train_test_split(train_data_sm, train_target_sm, test_size = .1, random_state = 0)

# Balance Data with Over-sampling
sm = SMOTE(ratio='auto',random_state = 0 , k=None, k_neighbors=5, m=None, m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=1)
x_train_data_balanced,y_train_data_balanced = sm.fit_sample(x_train, y_train.values.ravel())

# Shape from Balanced Data
x_train_data_balanced.shape
y_train_data_balanced.shape

# Practice your Balanced Data
practicedata = False

# ---------------------------------------------------------------------------------------------------------------

if practicedata == True:
    
    # Balanced Dataset
    
    # Naive Bayes with Balanced Data
    clf_balanced_GaussianNB = GaussianNB()
    clf_balanced_GaussianNB.fit(x_train_data_balanced,y_train_data_balanced)
    y_predict_GaussianNB = clf_balanced_GaussianNB.predict(x_val)
    y_true_GaussianNB = y_val
    
    # Evaluation of GaussianNB with Balanced Data
    print (classification_report(y_true_GaussianNB, y_predict_GaussianNB))
    print ("Confusion Matrix for GaussianNB with Balanced Data:")
    print(confusion_matrix(y_true_GaussianNB, y_predict_GaussianNB))
    print ("AUC score for GaussianNB with Balanced Data:")
    print(roc_auc_score(y_true_GaussianNB, y_predict_GaussianNB))
    
    # Support Vector Machine with Balanced Data
    clf_balanced_SVC = SVC(max_iter = 100,random_state=0)
    clf_balanced_SVC.fit(x_train_data_balanced,y_train_data_balanced)
    y_predict_SVC = clf_balanced_SVC.predict(x_val)
    y_true_SVC = y_val
    
    # Evaluation of SVC with Balanced Data
    print (classification_report(y_true_SVC, y_predict_SVC))
    print ("Confusion Matrix for SVC with Balanced Data:")
    print(confusion_matrix(y_true_SVC, y_predict_SVC))
    print ("AUC score for SVC with Balanced Data:")
    print(roc_auc_score(y_true_SVC, y_predict_SVC))
    
    # Linear Model with Balanced Data
    clf_balanced_SGDClassifier = SGDClassifier(max_iter=5,random_state=0)
    clf_balanced_SGDClassifier.fit(x_train_data_balanced,y_train_data_balanced)
    y_predict_SGDClassifier = clf_balanced_SGDClassifier.predict(x_val)
    y_true_SGDClassifier = y_val
    
    # Evaluation of Linear Model with Balanced Data
    print (classification_report(y_true_SGDClassifier, y_predict_SGDClassifier))
    print ("Confusion Matrix for SGDClassifier with Balanced Data:")
    print(confusion_matrix(y_true_SGDClassifier, y_predict_SGDClassifier))
    print ("AUC score for SGDClassifier with Balanced Data:")
    print(roc_auc_score(y_true_SGDClassifier, y_predict_SGDClassifier))
    
    # Random Forest with Balanced Data
    clf_balanced_RandomForestClassifier = RandomForestClassifier(random_state=0)
    clf_balanced_RandomForestClassifier.fit(x_train_data_balanced,y_train_data_balanced)
    y_predict_RandomForestClassifier = clf_balanced_RandomForestClassifier.predict(x_val)
    y_true_RandomForestClassifier = y_val
    
    # Evaluation of Random Forest with Balanced Data
    print (classification_report(y_true_RandomForestClassifier, y_predict_RandomForestClassifier))
    print ("Confusion Matrix for RandomForestClassifier with Balanced Data:")
    print(confusion_matrix(y_true_RandomForestClassifier, y_predict_RandomForestClassifier))
    print ("AUC score for RandomForestClassifier with Balanced Data:")
    print(roc_auc_score(y_true_RandomForestClassifier, y_predict_RandomForestClassifier))
    
# ---------------------------------------------------------------------------------------------------------------
    
    # Personal Information only with Balanced
    # Datas: x_train_data_balanced, y_train_data_balanced,x_val,y_val
    # Personal information - Columns 0-17
    
    x_train_data_balanced_ind = x_train_data_balanced[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
    y_train_data_balanced_ind = y_train_data_balanced
    x_val_ind = x_val[['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin']]
    y_val_ind = y_val
    
    # Naive Bayes with Balanced Data - Personal Information Only
    clf_balanced_GaussianNB_ind = GaussianNB()
    clf_balanced_GaussianNB_ind.fit(x_train_data_balanced_ind,y_train_data_balanced_ind)
    y_predict_GaussianNB_ind = clf_balanced_GaussianNB_ind.predict(x_val_ind)
    y_true_GaussianNB_ind = y_val_ind
    
    # Evaluation of GaussianNB with Balanced Data - Personal Information Only
    print (classification_report(y_true_GaussianNB_ind, y_predict_GaussianNB_ind))
    print ("Confusion Matrix for GaussianNB with Balanced Data - Personal Information Only:")
    print(confusion_matrix(y_true_GaussianNB_ind, y_predict_GaussianNB_ind))
    print ("AUC score for GaussianNB with Balanced Data - Personal Information Only:")
    print(roc_auc_score(y_true_GaussianNB_ind, y_predict_GaussianNB_ind))
    
    # Support Vector Machine with Balanced Data - Personal Information Only
    clf_balanced_SVC_ind = SVC(max_iter = 100,random_state=0)
    clf_balanced_SVC_ind.fit(x_train_data_balanced_ind,y_train_data_balanced_ind)
    y_predict_SVC_ind = clf_balanced_SVC_ind.predict(x_val_ind)
    y_true_SVC_ind = y_val
    
    # Evaluation of SVC with Balanced Data - Personal Information Only
    print (classification_report(y_true_SVC_ind, y_predict_SVC_ind))
    print ("Confusion Matrix for SVC with Balanced Data - Personal Information Only:")
    print(confusion_matrix(y_true_SVC_ind, y_predict_SVC_ind))
    print ("AUC score for SVC with Balanced Data - Personal Information Only:")
    print(roc_auc_score(y_true_SVC_ind, y_predict_SVC_ind))
    
    # Linear Model with Balanced Data - Personal Information Only
    clf_balanced_SGDClassifier_ind = SGDClassifier(max_iter=5,random_state=0)
    clf_balanced_SGDClassifier_ind.fit(x_train_data_balanced_ind,y_train_data_balanced_ind)
    y_predict_SGDClassifier_ind = clf_balanced_SGDClassifier_ind.predict(x_val_ind)
    y_true_SGDClassifier_ind = y_val
    
    # Evaluation of Linear Model with Balanced Data - Personal Information Only
    print (classification_report(y_true_SGDClassifier_ind, y_predict_SGDClassifier_ind))
    print ("Confusion Matrix for SGDClassifier with Balanced Data - Personal Information Only:")
    print(confusion_matrix(y_true_SGDClassifier_ind, y_predict_SGDClassifier_ind))
    print ("AUC score for SGDClassifier with Balanced Data - Personal Information Only:")
    print(roc_auc_score(y_true_SGDClassifier_ind, y_predict_SGDClassifier_ind))
    
    # Random Forest with Balanced Data - Personal Information Only
    clf_balanced_RandomForestClassifier_ind = RandomForestClassifier(random_state=0)
    clf_balanced_RandomForestClassifier_ind.fit(x_train_data_balanced_ind,y_train_data_balanced_ind)
    y_predict_RandomForestClassifier_ind = clf_balanced_RandomForestClassifier_ind.predict(x_val_ind)
    y_true_RandomForestClassifier_ind = y_val
    
    # Evaluation of Random Forest with Balanced Data - Personal Information Only
    print (classification_report(y_true_RandomForestClassifier_ind, y_predict_RandomForestClassifier_ind))
    print ("Confusion Matrix for RandomForestClassifier with Balanced Data - Personal Information Only:")
    print(confusion_matrix(y_true_RandomForestClassifier_ind, y_predict_RandomForestClassifier_ind))
    print ("AUC score for RandomForestClassifier with Balanced Data - Personal Information Only:")
    print(roc_auc_score(y_true_RandomForestClassifier_ind, y_predict_RandomForestClassifier_ind))
    
    # Region Information only with Balanced
    # Datas: x_train_data_balanced, y_train_data_balanced,x_val,y_val
    # Personal information - Columns 18-20
    
    x_train_data_balanced_reg = x_train_data_balanced[:,[18,19,20]]
    y_train_data_balanced_reg = y_train_data_balanced
    x_val_reg = x_val[['ps_reg_01','ps_reg_02','ps_reg_03']]
    y_val_reg = y_val
    
    # Naive Bayes with Balanced Data - Region Information Only
    clf_balanced_GaussianNB_reg = GaussianNB()
    clf_balanced_GaussianNB_reg.fit(x_train_data_balanced_reg,y_train_data_balanced_reg)
    y_predict_GaussianNB_reg = clf_balanced_GaussianNB_reg.predict(x_val_reg)
    y_true_GaussianNB_reg = y_val_reg
    
    # Evaluation of GaussianNB with Balanced Data - Region Information Only
    print (classification_report(y_true_GaussianNB_reg, y_predict_GaussianNB_reg))
    print ("Confusion Matrix for GaussianNB with Balanced Data - Region Information Only:")
    print(confusion_matrix(y_true_GaussianNB_reg, y_predict_GaussianNB_reg))
    print ("AUC score for GaussianNB with Balanced Data - Region Information Only:")
    print(roc_auc_score(y_true_GaussianNB_reg, y_predict_GaussianNB_reg))
    
    # Support Vector Machine with Balanced Data - Region Information Only
    clf_balanced_SVC_reg = SVC(max_iter = 100,random_state=0)
    clf_balanced_SVC_reg.fit(x_train_data_balanced_reg,y_train_data_balanced_reg)
    y_predict_SVC_reg = clf_balanced_SVC_reg.predict(x_val_reg)
    y_true_SVC_reg = y_val
    
    # Evaluation of SVC with Balanced Data - Region Information Only
    print (classification_report(y_true_SVC_reg, y_predict_SVC_reg))
    print ("Confusion Matrix for SVC with Balanced Data - Region Information Only:")
    print(confusion_matrix(y_true_SVC_reg, y_predict_SVC_reg))
    print ("AUC score for SVC with Balanced Data - Region Information Only:")
    print(roc_auc_score(y_true_SVC_reg, y_predict_SVC_reg))
    
    # Linear Model with Balanced Data - Region Information Only
    clf_balanced_SGDClassifier_reg = SGDClassifier(max_iter=5,random_state=0)
    clf_balanced_SGDClassifier_reg.fit(x_train_data_balanced_reg,y_train_data_balanced_reg)
    y_predict_SGDClassifier_reg = clf_balanced_SGDClassifier_reg.predict(x_val_reg)
    y_true_SGDClassifier_reg = y_val
    
    # Evaluation of Linear Model with Balanced Data - Region Information Only
    print (classification_report(y_true_SGDClassifier_reg, y_predict_SGDClassifier_reg))
    print ("Confusion Matrix for SGDClassifier with Balanced Data - Region Information Only:")
    print(confusion_matrix(y_true_SGDClassifier_reg, y_predict_SGDClassifier_reg))
    print ("AUC score for SGDClassifier with Balanced Data - Region Information Only:")
    print(roc_auc_score(y_true_SGDClassifier_reg, y_predict_SGDClassifier_reg))
    
    # Random Forest with Balanced Data - Region Information Only
    clf_balanced_RandomForestClassifier_reg = RandomForestClassifier(random_state=0)
    clf_balanced_RandomForestClassifier_reg.fit(x_train_data_balanced_reg,y_train_data_balanced_reg)
    y_predict_RandomForestClassifier_reg = clf_balanced_RandomForestClassifier_reg.predict(x_val_reg)
    y_true_RandomForestClassifier_reg = y_val
    
    # Evaluation of Random Forest with Balanced Data - Region Information Only
    print (classification_report(y_true_RandomForestClassifier_reg, y_predict_RandomForestClassifier_reg))
    print ("Confusion Matrix for RandomForestClassifier with Balanced Data - Region Information Only:")
    print(confusion_matrix(y_true_RandomForestClassifier_reg, y_predict_RandomForestClassifier_reg))
    print ("AUC score for RandomForestClassifier with Balanced Data - Region Information Only:")
    print(roc_auc_score(y_true_RandomForestClassifier_reg, y_predict_RandomForestClassifier_reg))
    
    # Auto Information only with Balanced
    # Datas: x_train_data_balanced, y_train_data_balanced,x_val,y_val
    # Personal information - Columns 21-36
    
    x_train_data_balanced_auto = x_train_data_balanced[:,[21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]]
    y_train_data_balanced_auto = y_train_data_balanced
    x_val_auto = x_val[['ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15']]
    y_val_auto = y_val
    
    # Naive Bayes with Balanced Data - Auto Information Only
    clf_balanced_GaussianNB_auto = GaussianNB()
    clf_balanced_GaussianNB_auto.fit(x_train_data_balanced_auto,y_train_data_balanced_auto)
    y_predict_GaussianNB_auto = clf_balanced_GaussianNB_auto.predict(x_val_auto)
    y_true_GaussianNB_auto = y_val_reg
    
    # Evaluation of GaussianNB with Balanced Data - Auto Information Only
    print (classification_report(y_true_GaussianNB_auto, y_predict_GaussianNB_auto))
    print ("Confusion Matrix for GaussianNB with Balanced Data - Auto Information Only:")
    print(confusion_matrix(y_true_GaussianNB_auto, y_predict_GaussianNB_auto))
    print ("AUC score for GaussianNB with Balanced Data - Auto Information Only:")
    print(roc_auc_score(y_true_GaussianNB_auto, y_predict_GaussianNB_auto))
    
    # Support Vector Machine with Balanced Data - Auto Information Only
    clf_balanced_SVC_auto = SVC(max_iter = 100,random_state=0)
    clf_balanced_SVC_auto.fit(x_train_data_balanced_auto,y_train_data_balanced_auto)
    y_predict_SVC_auto = clf_balanced_SVC_auto.predict(x_val_auto)
    y_true_SVC_auto = y_val
    
    # Evaluation of SVC with Balanced Data - Auto Information Only
    print (classification_report(y_true_SVC_auto, y_predict_SVC_auto))
    print ("Confusion Matrix for SVC with Balanced Data - Auto Information Only:")
    print(confusion_matrix(y_true_SVC_auto, y_predict_SVC_auto))
    print ("AUC score for SVC with Balanced Data - Auto Information Only:")
    print(roc_auc_score(y_true_SVC_auto, y_predict_SVC_auto))
    
    # Linear Model with Balanced Data - Auto Information Only
    clf_balanced_SGDClassifier_auto = SGDClassifier(max_iter=5,random_state=0)
    clf_balanced_SGDClassifier_auto.fit(x_train_data_balanced_auto,y_train_data_balanced_auto)
    y_predict_SGDClassifier_auto = clf_balanced_SGDClassifier_auto.predict(x_val_auto)
    y_true_SGDClassifier_auto = y_val
    
    # Evaluation of Linear Model with Balanced Data - Auto Information Only
    print (classification_report(y_true_SGDClassifier_auto, y_predict_SGDClassifier_auto))
    print ("Confusion Matrix for SGDClassifier with Balanced Data - Auto Information Only:")
    print(confusion_matrix(y_true_SGDClassifier_auto, y_predict_SGDClassifier_auto))
    print ("AUC score for SGDClassifier with Balanced Data - Auto Information Only:")
    print(roc_auc_score(y_true_SGDClassifier_auto, y_predict_SGDClassifier_auto))
    
    # Random Forest with Balanced Data - Auto Information Only
    clf_balanced_RandomForestClassifier_auto = RandomForestClassifier(random_state=0)
    clf_balanced_RandomForestClassifier_auto.fit(x_train_data_balanced_auto,y_train_data_balanced_auto)
    y_predict_RandomForestClassifier_auto = clf_balanced_RandomForestClassifier_auto.predict(x_val_auto)
    y_true_RandomForestClassifier_auto = y_val
    
    # Evaluation of Random Forest with Balanced Data - Auto Information Only
    print (classification_report(y_true_RandomForestClassifier_auto, y_predict_RandomForestClassifier_auto))
    print ("Confusion Matrix for RandomForestClassifier with Balanced Data - Auto Information Only:")
    print(confusion_matrix(y_true_RandomForestClassifier_auto, y_predict_RandomForestClassifier_auto))
    print ("AUC score for RandomForestClassifier with Balanced Data - Auto Information Only:")
    print(roc_auc_score(y_true_RandomForestClassifier_auto, y_predict_RandomForestClassifier_auto))
    
    # Calc Information only with Balanced
    # Datas: x_train_data_balanced, y_train_data_balanced,x_val,y_val
    # Personal information - Columns 37-56
    
    x_train_data_balanced_calc = x_train_data_balanced[:,[37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56]]
    y_train_data_balanced_calc = y_train_data_balanced
    x_val_calc = x_val[['ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']]
    y_val_calc = y_val
    
    # Naive Bayes with Balanced Data - Calc Information Only
    clf_balanced_GaussianNB_calc = GaussianNB()
    clf_balanced_GaussianNB_calc.fit(x_train_data_balanced_calc,y_train_data_balanced_calc)
    y_predict_GaussianNB_calc = clf_balanced_GaussianNB_calc.predict(x_val_calc)
    y_true_GaussianNB_calc = y_val_reg
    
    # Evaluation of GaussianNB with Balanced Data - Calc Information Only
    print (classification_report(y_true_GaussianNB_calc, y_predict_GaussianNB_calc))
    print ("Confusion Matrix for GaussianNB with Balanced Data - Calc Information Only:")
    print(confusion_matrix(y_true_GaussianNB_calc, y_predict_GaussianNB_calc))
    print ("AUC score for GaussianNB with Balanced Data - Calc Information Only:")
    print(roc_auc_score(y_true_GaussianNB_calc, y_predict_GaussianNB_calc))
    
    # Support Vector Machine with Balanced Data - Calc Information Only
    clf_balanced_SVC_calc = SVC(max_iter = 100,random_state=0)
    clf_balanced_SVC_calc.fit(x_train_data_balanced_calc,y_train_data_balanced_calc)
    y_predict_SVC_calc = clf_balanced_SVC_calc.predict(x_val_calc)
    y_true_SVC_calc = y_val
    
    # Evaluation of SVC with Balanced Data - Calc Information Only
    print (classification_report(y_true_SVC_calc, y_predict_SVC_calc))
    print ("Confusion Matrix for SVC with Balanced Data - Calc Information Only:")
    print(confusion_matrix(y_true_SVC_calc, y_predict_SVC_calc))
    print ("AUC score for SVC with Balanced Data - Calc Information Only:")
    print(roc_auc_score(y_true_SVC_calc, y_predict_SVC_calc))
    
    # Linear Model with Balanced Data - Calc Information Only
    clf_balanced_SGDClassifier_calc = SGDClassifier(max_iter=5,random_state=0)
    clf_balanced_SGDClassifier_calc.fit(x_train_data_balanced_calc,y_train_data_balanced_calc)
    y_predict_SGDClassifier_calc = clf_balanced_SGDClassifier_calc.predict(x_val_calc)
    y_true_SGDClassifier_calc = y_val
    
    # Evaluation of Linear Model with Balanced Data - Calc Information Only
    print (classification_report(y_true_SGDClassifier_calc, y_predict_SGDClassifier_calc))
    print ("Confusion Matrix for SGDClassifier with Balanced Data - Calc Information Only:")
    print(confusion_matrix(y_true_SGDClassifier_calc, y_predict_SGDClassifier_calc))
    print ("AUC score for SGDClassifier with Balanced Data - Calc Information Only:")
    print(roc_auc_score(y_true_SGDClassifier_calc, y_predict_SGDClassifier_calc))
    
    # Random Forest with Balanced Data - Calc Information Only
    clf_balanced_RandomForestClassifier_calc = RandomForestClassifier(random_state=0)
    clf_balanced_RandomForestClassifier_calc.fit(x_train_data_balanced_calc,y_train_data_balanced_calc)
    y_predict_RandomForestClassifier_calc = clf_balanced_RandomForestClassifier_calc.predict(x_val_calc)
    y_true_RandomForestClassifier_calc = y_val
    
    # Evaluation of Random Forest with Balanced Data - Calc Information Only
    print (classification_report(y_true_RandomForestClassifier_calc, y_predict_RandomForestClassifier_calc))
    print ("Confusion Matrix for RandomForestClassifier with Balanced Data - Calc Information Only:")
    print(confusion_matrix(y_true_RandomForestClassifier_calc, y_predict_RandomForestClassifier_calc))
    print ("AUC score for RandomForestClassifier with Balanced Data - Calc Information Only:")
    print(roc_auc_score(y_true_RandomForestClassifier_calc, y_predict_RandomForestClassifier_calc))
    
    # ---------------------------------------------------------------------------------------------------------------
    
    # Removing all Columns with Nan values:
    # train_data.isnull().any()
    # Columns: ps_ind_02_cat, ps_ind_04_cat, ps_ind_05_cat, ps_reg_03, ps_car_01_cat, ps_car_02_cat, ps_car_03_cat, ps_car_05_cat, ps_car_07_cat, ps_car_09_cat, ps_car_11, ps_car_12, ps_car_14
    # Columns Number: 2,4,5,21,22,23,24,26,28,30,33,34,36
    
    # ID has already been dropped, Columns with Nan Values are removed
    test_data_all = test_data.drop(['id','ps_ind_02_cat','ps_ind_04_cat', 'ps_ind_05_cat', 'ps_reg_03', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_05_cat', 'ps_car_07_cat', 'ps_car_09_cat', 'ps_car_11', 'ps_car_12', 'ps_car_14'],axis=1)
    x_train_data_balanced_all = x_train_data_balanced[:,[0,1,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,27,29,31,32,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56]]
    y_train_data_balanced_all = y_train_data_balanced
    
    # Naive Bayes - Trained with Balanced Dataset
    clf_all_GaussianNB = GaussianNB()
    clf_all_GaussianNB.fit(x_train_data_balanced_all,y_train_data_balanced_all)
    result_all_GaussianNB = clf_all_GaussianNB.predict_proba(test_data_all)
    
    # Support Vector Machine - Trained with Balanced Dataset
    clf_all_SVC = SVC(max_iter = 100,probability=True)
    clf_all_SVC.fit(x_train_data_balanced_all,y_train_data_balanced_all)
    result_all_SVC = clf_all_SVC.predict_proba(test_data_all)
    
    # Linear Model - Trained with Balanced Dataset
    clf_all_SGDClassifier = SGDClassifier(max_iter=5,loss = 'log')
    clf_all_SGDClassifier.fit(x_train_data_balanced_all,y_train_data_balanced_all)
    result_all_SGDClassifier = clf_all_SGDClassifier.predict_proba(test_data_all)
    
    # Random Forest - Trained with Balanced Dataset
    clf_all_RandomForestClassifier = RandomForestClassifier()
    clf_all_RandomForestClassifier.fit(x_train_data_balanced_all,y_train_data_balanced_all)
    result_all_RandomForestClassifier = clf_all_RandomForestClassifier.predict_proba(test_data_all)
else:
    print('No Practice has been done')
# ---------------------------------------------------------------------------------------------------------------

# Want to score with different methods
scorediffm = False

if scorediffm == True:

    # Scoring with different Methods 
    
    if val_nan == True:
        ### Only Personal Information with Nan Values
        x_ind = train_data.drop(['target','id','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
        y_ind = train_data[['target']]
        test_ind = test_data.drop(['id','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
    elif val_nan == False:
        ### Only Personal Information without Nan values
        x_ind = train_data_clean.drop(['target','id','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
        y_ind = train_data_clean[['target']]
        test_ind = test_data_clean.drop(['id','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
    else:
        print('val_nan got crazy!')
    
    # Naive Bayes with Personal Information only - Trained with Balanced Dataset
    clf_ind_GaussianNB = GaussianNB()
    clf_ind_GaussianNB.fit(x_train_data_balanced_ind,y_train_data_balanced_ind)
    result_ind_GaussianNB = clf_ind_GaussianNB.predict_proba(test_ind)
    
    # Support Vector Machine with Personal Information only - Trained with Balanced Dataset
    clf_ind_SVC = SVC(max_iter = 100,probability=True)
    clf_ind_SVC.fit(x_train_data_balanced_ind,y_train_data_balanced_ind)
    result_ind_SVC = clf_ind_SVC.predict_proba(test_ind)
    
    # Linear Model with Personal Information only - Trained with Balanced Dataset
    clf_ind_SGDClassifier = SGDClassifier(max_iter=5,loss = 'log')
    clf_ind_SGDClassifier.fit(x_train_data_balanced_ind,y_train_data_balanced_ind)
    result_ind_SGDClassifier = clf_ind_SGDClassifier.predict_proba(test_ind)
    
    # Random Forest with Personal Information only - Trained with Balanced Dataset
    clf_ind_RandomForestClassifier = RandomForestClassifier()
    clf_ind_RandomForestClassifier.fit(x_train_data_balanced_ind,y_train_data_balanced_ind)
    result_ind_RandomForestClassifier = clf_ind_RandomForestClassifier.predict_proba(test_ind)
    
    if val_nan == True:
        ### Only Region Information with Nan Values
        x_reg = train_data.drop(['target','id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
        y_reg = train_data[['target']]
        test_reg = test_data.drop(['id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
    elif val_nan == False:
        ### Only Region Information without Nan Values
        x_reg = train_data_clean.drop(['target','id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
        y_reg = train_data_clean[['target']]
        test_reg = test_data_clean.drop(['id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
    else:
        print('val_nan got crazy!')
        
    # Naive Bayes with Region Information only - Trained with Balanced Dataset
    clf_reg_GaussianNB = GaussianNB()
    clf_reg_GaussianNB.fit(x_train_data_balanced_reg,y_train_data_balanced_reg)
    result_reg_GaussianNB = clf_reg_GaussianNB.predict_proba(test_reg)
    
    # Linear Model with Region Information only - Trained with Balanced Dataset
    clf_reg_SGDClassifier = SGDClassifier(max_iter=5,loss = 'log')
    clf_reg_SGDClassifier.fit(x_train_data_balanced_reg,y_train_data_balanced_reg)
    result_reg_SGDClassifier = clf_reg_SGDClassifier.predict_proba(test_reg)
    
    # Random Forest with Region Information only - Trained with Balanced Dataset
    clf_reg_RandomForestClassifier = RandomForestClassifier()
    clf_reg_RandomForestClassifier.fit(x_train_data_balanced_reg,y_train_data_balanced_reg)
    result_reg_RandomForestClassifier = clf_reg_RandomForestClassifier.predict_proba(test_reg)
    
    # Support Vector Machine with Region Information only - Trained with Balanced Dataset
    clf_reg_SVC = SVC(max_iter = 100,probability=True)
    clf_reg_SVC.fit(x_train_data_balanced_reg,y_train_data_balanced_reg)
    result_reg_SVC = clf_reg_SVC.predict_proba(test_reg)
    
    if val_nan == True:
        ### Only Auto Information with Nan Values
        x_auto = train_data.drop(['target','id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
        y_auto = train_data[['target']]
        test_auto = test_data.drop(['id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
    elif val_nan == False:
        ### Only Auto Information without Nan Values
        x_auto = train_data_clean.drop(['target','id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
        y_auto = train_data_clean[['target']]
        test_auto = test_data_clean.drop(['id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
    else:
        print('val_nan got crazy!')
        
    # Naive Bayes with Auto Information only - Trained with Balanced Dataset
    clf_car_GaussianNB = GaussianNB()
    clf_car_GaussianNB.fit(x_train_data_balanced_auto,y_train_data_balanced_auto)
    result_car_GaussianNB = clf_car_GaussianNB.predict_proba(test_auto)
    
    # Linear Model with Auto Information only - Trained with Balanced Dataset
    clf_car_SGDClassifier = SGDClassifier(max_iter=5,loss = 'log')
    clf_car_SGDClassifier.fit(x_train_data_balanced_auto,y_train_data_balanced_auto)
    result_car_SGDClassifier = clf_car_SGDClassifier.predict_proba(test_auto)
    
    # Random Forest with Auto Information only - Trained with Balanced Dataset
    clf_car_RandomForestClassifier = RandomForestClassifier()
    clf_car_RandomForestClassifier.fit(x_train_data_balanced_auto,y_train_data_balanced_auto)
    result_car_RandomForestClassifier = clf_car_RandomForestClassifier.predict_proba(test_auto)
    
    # Support Vector Machine with Auto Information only - Trained with Balanced Dataset
    clf_car_SVC = SVC(max_iter = 100,probability=True)
    clf_car_SVC.fit(x_train_data_balanced_auto,y_train_data_balanced_auto)
    result_car_SVC = clf_car_SVC.predict_proba(test_auto)
    
    if val_nan == True:
        ### Only calculated Information with Nan Values
        x_calc = train_data.drop(['target','id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15'], axis =1)
        y_calc = train_data[['target']]
        test_calc = test_data.drop(['id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15'], axis =1)
    elif val_nan == False:
        ### Only calculated Information without Nan Values
        x_calc = train_data_clean.drop(['target','id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15'], axis =1)
        y_calc = train_data_clean[['target']]
        test_calc = test_data_clean.drop(['id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15'], axis =1)
    else:
        print('val_nan got crazy!')
        
    # Naive Bayes with Calculated Information only - Trained with Balanced Dataset
    clf_calc_GaussianNB = GaussianNB()
    clf_calc_GaussianNB.fit(x_train_data_balanced_calc,y_train_data_balanced_calc)
    result_calc_GaussianNB = clf_calc_GaussianNB.predict_proba(test_calc)
    
    # Linear Model with Calculated Information only - Trained with Balanced Dataset
    clf_calc_SGDClassifier = SGDClassifier(max_iter=5,loss = 'log')
    clf_calc_SGDClassifier.fit(x_train_data_balanced_calc,y_train_data_balanced_calc)
    result_calc_SGDClassifier = clf_calc_SGDClassifier.predict_proba(test_calc)
    
    # Random Forest with Calculated Information only - Trained with Balanced Dataset
    clf_calc_RandomForestClassifier = RandomForestClassifier()
    clf_calc_RandomForestClassifier.fit(x_train_data_balanced_calc,y_train_data_balanced_calc)
    result_calc_RandomForestClassifier = clf_calc_RandomForestClassifier.predict_proba(test_calc)
    
    # Support Vector Machine with Calculated Information only - Trained with Balanced Dataset
    clf_calc_SVC = SVC(max_iter = 100,probability=True)
    clf_calc_SVC.fit(x_train_data_balanced_calc,y_train_data_balanced_calc)
    result_calc_SVC = clf_calc_SVC.predict_proba(test_calc)
else:
    print('No score with different methods')
# ---------------------------------------------------------------------------------------------------------------

# Data Selection - Features

# Import the Feature Selection Library
from sklearn.feature_selection import SelectKBest, VarianceThreshold, SelectPercentile, chi2, f_classif

# Split train_data without missing values
x_train_data_selection = train_data_clean.drop(['id','target'], axis = 1)
y_train_data_selection = train_data_clean[['target']]

# New Array of Features without ID and Target
features_train_selection = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']
features_test_selection = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']

# Threshold for VarianceThreshold
thresholdforpreprocessing = 0.1

# Apply VarianceThreshold to train_data
selector = VarianceThreshold(threshold=thresholdforpreprocessing)
variance_train_data = selector.fit_transform(x_train_data_selection,y_train_data_selection)

# Get Features from train_data - VarianceThreshold
variance_array = selector.get_support()

# Create list for feaatures that are removed through VarianceThreshold
variance_values_removed = []
variance_values_maintained = []

# List to Store the Features that are removed
for i in range(len(variance_array)):
    if variance_array[i] == False:
        variance_values_removed.insert(i,features_train_selection[i])
    else:
        variance_values_maintained.insert(i,features_train_selection[i])

# Features removed through VarianceThreshold
print('Features removed through VarianceThreshold:')
print(variance_values_removed)

# Amount of features Removed
len(variance_values_removed)

#Features maintained
print('Features maintained:')
print(variance_values_maintained)

# Amount of features Maintained
len(variance_values_maintained)

# ---------------------------------------------------------------------------------------------------------------

# Percentage of Features to keep (%)
select_percentile = 50

# Amount of features after SelectPercentile
percentile_features_maintained = int(0.5*len(variance_values_maintained))
print('Amount of features maintained after select percentile')
print(percentile_features_maintained)
print('Percentage of Features Removed:')
print(select_percentile)

# Drop the Features from VarianceThreshold
x_train_data_selectpercentile = x_train_data_selection.drop(variance_values_removed, axis=1)
y_train_data_selectpercentile = y_train_data_selection

# Apply SelectPercentile to train data - Evaluation chi2
percentile = SelectPercentile(chi2, percentile = select_percentile)
percentile_train_data = percentile.fit(x_train_data_selectpercentile,y_train_data_selectpercentile)

# Get Features from train_data_selection - SelectPercentile
percentile_array = percentile_train_data.get_support()

# Create list for feaatures that are removed through SelectPercentile
percentile_values_removed = []
percentile_values_maintained = []

# List to Store the Features that are removed
for i in range(len(percentile_array)):
    if percentile_array[i] == False:
        percentile_values_removed.insert(i,variance_values_maintained[i])
    else:
        percentile_values_maintained.insert(i,variance_values_maintained[i])

# Features removed through SelectPercentile
print('Features removed through SelectPercentile:')
print(percentile_values_removed)

# Amount of features Removed
len(percentile_values_removed)

#Features maintained
print('Features maintained:')
print(percentile_values_maintained)

# Amount of features Maintained
len(percentile_values_maintained)

# ---------------------------------------------------------------------------------------------------------------

# Amount of Features to keep - SelectKBest
kbest_features = int(0.8*(len(percentile_values_maintained)))

# Amount of features after K Best
print('Amount of features maintained after KBest')
print(kbest_features)

# Drop the Features from SelectPercentile
x_train_data_kbest = x_train_data_selectpercentile.drop(percentile_values_removed, axis=1)
y_train_data_kbest = y_train_data_selectpercentile

# Apply SelectKBest to data - Evaluation f_classif
selectK = SelectKBest(f_classif, k=kbest_features)
selectK_train_data = selectK.fit(x_train_data_kbest,y_train_data_kbest)

# Get Features from train_data_percentile - SelectKBest
selectK_array = selectK_train_data.get_support()

# Create list for feaatures that are removed through SelectKBest
selectK_values_removed = []
selectK_values_maintained = []

# List to Store the Features that are removed
for i in range(len(selectK_array)):
    if selectK_array[i] == False:
        selectK_values_removed.insert(i,percentile_values_maintained[i])
    else:
        selectK_values_maintained.insert(i,percentile_values_maintained[i])

# Features removed through SelectKBest
print('Features removed through SelectKBest:')
print(selectK_values_removed)

# Amount of features Removed
len(selectK_values_removed)

#Features maintained
print('Features maintained:')
print(selectK_values_maintained)

# Amount of features Maintained
len(selectK_values_maintained)

# ---------------------------------------------------------------------------------------------------------------

# Scoring with Features Selected - Best Only

if val_nan == True:
    ### Best Features Only
    ## First Set of Features
    #(Removed Auto Information and Calculated Information) with Nan Values
    x_best = train_data[selectK_values_maintained]
    y_best = train_data[['target']]
    test_best = test_data[selectK_values_maintained]
elif val_nan == False:
    #(Removed Auto Information and Calculated Information) without Nan Values
    x_best = train_data_clean[selectK_values_maintained]
    y_best = train_data_clean[['target']]
    test_best = test_data_clean[selectK_values_maintained]
else:
    print('val_nan got crazy!')

# Imput Missing Values
missingvaluesimput = False

#Imputing missing values for both train and test
if missingvaluesimput == True:
    x_best.fillna(-999, inplace=True)
    test_best.fillna(-999,inplace=True)
else:
    print('No Imput Values has been made')

# Use the 4 Techniques
    techuse = False
    
if techuse == True:
    
    # Naive Bayes with Best Information only
    clf_best_GaussianNB = GaussianNB()
    clf_best_GaussianNB.fit(x_best,y_best)
    result_best_GaussianNB = clf_best_GaussianNB.predict_proba(test_best)
    
    # Linear Model with Best Information only
    clf_best_SGDClassifier = SGDClassifier(max_iter=5,loss = 'log')
    clf_best_SGDClassifier.fit(x_best,y_best)
    result_best_SGDClassifier = clf_best_SGDClassifier.predict_proba(test_best)
    
    # Random Forest with Best Information only
    clf_best_RandomForestClassifier = RandomForestClassifier()
    clf_best_RandomForestClassifier.fit(x_best,y_best)
    result_best_RandomForestClassifier = clf_best_RandomForestClassifier.predict_proba(test_best)
    
    # Support Vector Machine with Best Information only
    clf_best_SVC = SVC(max_iter = 100,probability=True)
    clf_best_SVC.fit(x_best,y_best)
    result_best_SVC = clf_best_SVC.predict_proba(test_best)
else:
    print('4 techniques not used')    

# Try outs with another techniques - Best Features only
# Import Library
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Use others techniques
otherstechuse = False

if otherstechuse == True:

    # Decision Tree Regression with best Information Only
    clf_best_DecisionTreeClassifier = DecisionTreeClassifier()
    clf_best_DecisionTreeClassifier.fit(x_best,y_best)
    result_best_DecisionTreeClassifier = clf_best_DecisionTreeClassifier.predict_proba(test_best)
    
    # AdaBoost Regressor with best Information only
    clf_best_AdaBoostClassifier = AdaBoostClassifier()
    clf_best_AdaBoostClassifier.fit(x_best,y_best)
    result_best_AdaBoostClassifier = clf_best_AdaBoostClassifier.predict_proba(test_best)
    
    # Gaussian Process Classifier with best Information only
    clf_best_GaussianProcessClassifier = GaussianProcessClassifier()
    clf_best_GaussianProcessClassifier.fit(x_best,y_best)
    result_best_GaussianProcessClassifier = clf_best_GaussianProcessClassifier.predict_proba(test_best)
    
    # K Nearest Neighboor Regressor with best Information only
    clf_best_KNeighborsClassifier = KNeighborsClassifier()
    clf_best_KNeighborsClassifier.fit(x_best,y_best)
    result_best_KNeighborsClassifier = clf_best_KNeighborsClassifier.predict_proba(test_best)
    
    # Passive Aggressive Regressor with best Information only
    clf_best_PassiveAggressiveClassifier = PassiveAggressiveClassifier()
    clf_best_PassiveAggressiveClassifier.fit(x_best,y_best)
    result_best_PassiveAggressiveClassifier = clf_best_PassiveAggressiveClassifier.predict_proba(test_best)
else:
    print('No other techniques has been used')
    
# The best 2 algorithms

# Wanna use CatBoost
catboostuse = False

if catboostuse == True:

    # Cat Boost Regressor with best Information only
    clf_best_CatBoostClassifier = CatBoostClassifier()
    clf_best_CatBoostClassifier.fit(x_best,y_best)
    result_best_CatBoostClassifier = clf_best_CatBoostClassifier.predict_proba(test_best)
else:
    print ('Cat Boost has not been used')
    
# Wanna use XGB
xgboostuse = True

if xgboostuse == True:
    
    # XGBoost with best Information only
    clf_best_XGB = XGBClassifier()
    clf_best_XGB.fit(x_best,y_best)
    result_best_XGB = clf_best_XGB.predict_proba(test_best)
else:
    print('XGB has not been used')

# Evaluate algorithms with best features
best_training = True

if best_training == True:
    x_train_data_balanced_pre = pd.DataFrame(data = x_train_data_balanced, columns = features_train_selection)
    x_best_2 = x_train_data_balanced_pre[selectK_values_maintained]
    y_best_2 = y_train_data_balanced
    test_best_2 = y_val
    test_best_3 = x_val[selectK_values_maintained]
    
    # Naive Bayes with Balanced Data - - Best Features
    clf_balanced_GaussianNB_best = GaussianNB()
    clf_balanced_GaussianNB_best.fit(x_best_2,y_best_2)
    y_predict_GaussianNB_best = clf_balanced_GaussianNB_best.predict(test_best_3)
    y_true_GaussianNB_best = test_best_2
    
    # Evaluation of GaussianNB with Balanced Data - Best Features
    print (classification_report(y_true_GaussianNB_best, y_predict_GaussianNB_best))
    print ("Confusion Matrix for GaussianNB with Balanced Data - Best Features:")
    print(confusion_matrix(y_true_GaussianNB_best, y_predict_GaussianNB_best))
    print ("AUC score for GaussianNB with Balanced Data - Best Features:")
    print(roc_auc_score(y_true_GaussianNB_best, y_predict_GaussianNB_best))
    
    # Support Vector Machine with Balanced Data - - Best Features
    clf_balanced_SVC_best = SVC(max_iter = 100,random_state=0)
    clf_balanced_SVC_best.fit(x_best_2,y_best_2)
    y_predict_SVC_best = clf_balanced_SVC_best.predict(test_best_3)
    y_true_SVC_best = test_best_2
    
    # Evaluation of SVC with Balanced Data - Best Features
    print (classification_report(y_true_SVC_best, y_predict_SVC_best))
    print ("Confusion Matrix for SVC with Balanced Data - Best Features:")
    print(confusion_matrix(y_true_SVC_best, y_predict_SVC_best))
    print ("AUC score for SVC with Balanced Data - Best Features:")
    print(roc_auc_score(y_true_SVC_best, y_predict_SVC_best))
    
    # Linear Model with Balanced Data - Best Features
    clf_balanced_SGDClassifier_best = SGDClassifier(max_iter=5,random_state=0)
    clf_balanced_SGDClassifier_best.fit(x_best_2,y_best_2)
    y_predict_SGDClassifier_best = clf_balanced_SGDClassifier_best.predict(test_best_3)
    y_true_SGDClassifier_best = test_best_2
    
    # Evaluation of Linear Model with Balanced Data - Best Features
    print (classification_report(y_true_SGDClassifier_best, y_predict_SGDClassifier_best))
    print ("Confusion Matrix for SGDClassifier with Balanced Data - Best Features:")
    print(confusion_matrix(y_true_SGDClassifier_best, y_predict_SGDClassifier_best))
    print ("AUC score for SGDClassifier with Balanced Data - Best Features:")
    print(roc_auc_score(y_true_SGDClassifier_best, y_predict_SGDClassifier_best))
    
    # Random Forest with Balanced Data - Best Features
    clf_balanced_RandomForestClassifier_best = RandomForestClassifier(random_state=0)
    clf_balanced_RandomForestClassifier_best.fit(x_best_2,y_best_2)
    y_predict_RandomForestClassifier_best = clf_balanced_RandomForestClassifier_best.predict(test_best_3)
    y_true_RandomForestClassifier_best = test_best_2
    
    # Evaluation of Random Forest with Balanced Data - Best Features
    print (classification_report(y_true_RandomForestClassifier_best, y_predict_RandomForestClassifier_best))
    print ("Confusion Matrix for RandomForestClassifier with Balanced Data - Best Features:")
    print(confusion_matrix(y_true_RandomForestClassifier_best, y_predict_RandomForestClassifier_best))
    print ("AUC score for RandomForestClassifier with Balanced Data - Best Features:")
    print(roc_auc_score(y_true_RandomForestClassifier_best, y_predict_RandomForestClassifier_best))
    
else:
    print('Best_training is False')
=======
import numpy as np
import pandas as pd

# Capstone Project Renan Hidani

# Only Personal Information
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

x_ind = train_data.drop(['target','id','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
y_ind = train_data[['target']]

# Only Region Information
train_data = pd.read_csv ('train.csv')
test_data = pd.read_csv('test.csv')

x_reg = train_data.drop(['target','id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
y_reg = y_ind

# Only Auto Information
train_data = pd.read_csv ('train.csv')
test_data = pd.read_csv('test.csv')

x_car = train_data.drop(['target','id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'], axis =1)
y_car = y_reg

# Only Calculated Information

train_data = pd.read_csv ('train.csv')
test_data = pd.read_csv('test.csv')

x_calc = train_data.drop(['target','id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15'], axis =1)
y_calc = y_car


print y_ind.count('target')
>>>>>>> origin/master
