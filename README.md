# README
## Author: Renan Hidani
### Udacity Capstone Machine Learning Nanodegree
#### Porto Seguro's Safe Driver Prediction Competition

##### Softwares

* Python 2.7.14 64 bits
* [Anaconda 5.0.1](https://anaconda.org/)
* Spyder 3.2.4
* Notebook 5.0.0

##### LIBRARIES

For Visualization (Bar graph and heatmap )
* [Seaborn](https://seaborn.pydata.org/)
* [PyPlot](https://matplotlib.org/)

For Visualization (Dendrogram and missing values)
* [missingno](https://github.com/ResidentMario/missingno)

* [Numpy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [XGBClassifier](http://xgboost.readthedocs.io/en/latest/)
* [CatBoostClassifier](https://catboost.yandex/)
* [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* [GaussianNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
* [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
* [PassiveAggressiveClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)
* [AdaBoostClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [GaussianProcessClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)
* [KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

For Preprocessing
* [SMOTE](http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html)

For Model Selection
* [Train Test Split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

For Metrics
* [Classification Report](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
* [Confusion Matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)
* [AUC](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)

For Feature Selection
* [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)
* [VarianceThreshold](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold)
* [SelectPercentile](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile)
* [chi2](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2)
* [f_classif](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif)

##### List of Variables

features_train - List of Features inside train.csv
features_test - List of Features inside test.csv
train_data - Data from train.csv
test_data - Data frmo test.csv
train_id - List of IDs from train.csv
test_id - List of IDs from test.csv
train_target - List of Target from train.csv
corr - Correlation Matrix
mask - Generation of upper triangle for heatmap
train_data_clean - Data from train_data without the missing values (-1)
test_data_clean - Data from test_data without the missing values (-1)
val_nan - Evaluation of algorithms with missing values values
train_data_sm - Data from train_data without missing values, dropping ID and Target
train_target_sm - Data from train_data without missing values, getting Target only
x_train - New Data without missing values for training, features without Target(split train test)
y_train - New Data without missing values for training, Target Only(split train test)
x_val - Evaluation of x_train
y_val - Evaluation of x_train
sm - Parameters for SMOTE
x_train_data_balanced - x_train, target balanced(SMOTE)
y_train_data_balanced - y_train, target balanced(SMOTE) - Target Values only
practicedata - Run GaussianNB, SVC, SGDClassifier, RandomForestClassifier for each group of feature(personal information, region information, car information, calculated informatuon,all features)
y_predict_GaussianNB - Prediction from GaussianNB with all features balanced
y_true_GaussianNB - True values from GaussianNB with all features balanced
y_predict_SVC - Prediction from SVC with all features balanced
y_true_SVC - True values from SVC with all features balanced
y_predict_SGDClassifier - Prediction from SGDClassifier with all features balanced
y_true_SGDClassifier - True values from SGDClassifier with all features balanced
y_predict_RandomForestClassifier - Prediction from RandomForestClassifier with all features balanced
y_true_RandomForestClassifier - True values from RandomForestClassifier with all features balanced
x_train_data_balanced_ind - x_train_balanced with personal information only (_ind_)
y_train_data_balanced_ind - same as y_train_balanced
x_val_ind - x_val with personal information only
y_val_ind - same as y_val
y_predict_GaussianNB_ind - Prediction from GaussianNB with personal features balanced
y_true_GaussianNB_ind - True values from GaussianNB with personal features balanced
y_predict_SVC_ind - Prediction from SVC with personal features balanced
y_true_SVC_ind - True values from SVC with personal features balanced
y_predict_SGDClassifier_ind - Prediction from SGDClassifier with personal features balanced
y_true_SGDClassifier_ind - True values from SGDClassifier with personal features balanced
y_predict_RandomForestClassifier_ind - Prediction from RandomForestClassifier with personal features balanced
y_true_RandomForestClassifier_ind - True values from RandomForestClassifier with personal features balanced
x_train_data_balanced_reg - x_train_balanced with region information only (_reg_)
y_train_data_balanced_reg - same as y_train_balanced
x_val_reg - x_val with region information only
y_val_reg - same as y_val_ind
y_predict_GaussianNB_reg - Prediction from GaussianNB with region features balanced
y_true_GaussianNB_reg - True values from GaussianNB with region features balanced
y_predict_SVC_reg - Prediction from SVC with region features balanced
y_true_SVC_reg - True values from SVC with region features balanced
y_predict_SGDClassifier_reg - Prediction from SGDClassifier with region features balanced
y_true_SGDClassifier_reg - True values from SGDClassifier with region features balanced
y_predict_RandomForestClassifier_reg - Prediction from RandomForestClassifier with region features balanced
y_true_RandomForestClassifier_reg - True values from RandomForestClassifier with region features balanced
x_train_data_balanced_auto - x_train_balanced with auto information only (_car_)
y_train_data_balanced_auto - same as y_train_balanced
x_val_auto - x_val with auto information only
y_val_auto - same as y_val_reg
y_predict_GaussianNB_auto - Prediction from GaussianNB with auto features balanced
y_true_GaussianNB_auto - True values from GaussianNB with auto features balanced
y_predict_SVC_auto - Prediction from SVC with auto features balanced
y_true_SVC_auto - True values from SVC with auto features balanced
y_predict_SGDClassifier_auto - Prediction from SGDClassifier with auto features balanced
y_true_SGDClassifier_auto - True values from SGDClassifier with auto features balanced
y_predict_RandomForestClassifier_auto - Prediction from RandomForestClassifier with auto features balanced
y_true_RandomForestClassifier_auto - True values from RandomForestClassifier with auto features balanced
x_train_data_balanced_calc - x_train_balanced with calculated information only (_calc_)
y_train_data_balanced_calc - same as y_train_balanced
x_val_calc - x_val with calculated information only
y_val_calc - same as y_val_auto
y_predict_GaussianNB_calc - Prediction from GaussianNB with calculated features balanced
y_true_GaussianNB_calc - True values from GaussianNB with calculated features balanced
y_predict_SVC_calc - Prediction from SVC with calculated features balanced
y_true_SVC_calc - True values from SVC with calculated features balanced
y_predict_SGDClassifier_calc - Prediction from SGDClassifier with calculated features balanced
y_true_SGDClassifier_calc - True values from SGDClassifier with calculated features balanced
y_predict_RandomForestClassifier_calc - Prediction from RandomForestClassifier with calculated features balanced
y_true_RandomForestClassifier_calc - True values from RandomForestClassifier with calculated features balanced
test_data_all - test_data without features with missing values
x_train_data_balanced_all - x_train_balanced without features with missing values
y_train_data_balanced_all - same as y_train_data_balanced
result_all_GaussianNB - Probability from GaussianNB with all features balanced without features with missing values
result_all_SVC - Probability from SVC with all features balanced without features with missing values
result_all_SGDClassifier - Probability from SGDClassifier with all features balanced without features with missing values
result_all_RandomForestClassifier - Probability from RandomForestClassifier with all features balanced without features with missing values
scorediffm - Run GaussianNB, SVC, SGDClassifier, RandomForestClassifier for each group of feature(personal information, region information, car information, calculated informatuon), with data **unbalanced**
x_ind - train_data with personal information only, data unbalanced, may contain missing values or not (val_nan makes the call)
y_ind - train_data with target only, data unbalanced
test_ind - test_data with personal information only, data unbalanced, may contain missing values or not (val_nan makes the call)
result_ind_GaussianNB - Probability from GaussianNB with personal features, target unbalanced
result_ind_SVC - Probability from SVC with personal features, target unbalanced
result_ind_SGDClassifier - Probability from SGDClassifier with personal features, target unbalanced
result_ind_RandomForestClassifier - Probability from RandomForestClassifier with personal features, target unbalanced
x_reg - train_data with region information only, data unbalanced, may contain missing values or not (val_nan makes the call)
y_reg - train_data with target only, data unbalanced
test_reg - test_data with region information only, data unbalanced, may contain missing values or not (val_nan makes the call)
result_reg_GaussianNB - Probability from GaussianNB with region features, target unbalanced
result_reg_SVC - Probability from SVC with region features, target unbalanced
result_reg_SGDClassifier - Probability from SGDClassifier with region features, target unbalanced
result_reg_RandomForestClassifier - Probability from RandomForestClassifier with region features, target unbalanced
x_auto - train_data with auto information only, data unbalanced, may contain missing values or not (val_nan makes the call)
y_auto - train_data with target only, data unbalanced
test_auto - test_data with auto information only, data unbalanced, may contain missing values or not (val_nan makes the call)
result_car_GaussianNB - Probability from GaussianNB with auto features, target unbalanced
result_car_SVC - Probability from SVC with auto features, target unbalanced
result_car_SGDClassifier - Probability from SGDClassifier with auto features, target unbalanced
result_car_RandomForestClassifier - Probability from RandomForestClassifier with auto features, target unbalanced
x_calc - train_data with calculated information only, data unbalanced, may contain missing values or not (val_nan makes the call)
y_calc - train_data with target only, data unbalanced
test_calculated - test_data with calculated information only, data unbalanced, may contain missing values or not (val_nan makes the call)
result_calc_GaussianNB - Probability from GaussianNB with calculated features, target unbalanced
result_calc_SVC - Probability from SVC with calculated features, target unbalanced
result_calc_SGDClassifier - Probability from SGDClassifier with calculated features, target unbalanced
result_calc_RandomForestClassifier - Probability from RandomForestClassifier with calculated features, target unbalanced
x_train_data_selection - train_data_clean, dropping ID and Target
y_train_data_selection - train_data_clean, getting target only
features_train_selection - features_train without ID and Target
features_test_selection - features_test without ID
thresholdforpreprocessing - Threshold for VarianceThreshold
variance_train_data - Features selected through VarianceThreshold
variance_array - Index for features selected through VarianceThreshold
variance_values_removed - Features removed through VarianceThreshold
variance_values_maintained - Featured maintained through VarianceThreshold
select_percentile - Percentage for SelectPercentile
percentile_features_maintained - Amount of features maintained after SelectPercentile
x_train_data_selectpercentile - x_train_data_selection, dropping features after VarianceThreshold
y_train_data_selectpercentile - same as y_train_data_selection
percentile_array - Index for features selected through SelectPercentile
percentile_values_removed - Features removed through SelectPercentile
percentile_values_maintained - Features maintained through SelectPercentile
kbest_features - Amount of features to keep after SelectKBest
x_train_data_kbest - x_train_data_selectpercentile, dropping features after SelectPercentile
y_train_data_kbest - same as y_train_data_selectpercentile
selectK_array - Index for features selected through SelectKBest
selectK_values_removed - Features removed through SelectKBest
selectK_values_maintained - Features maintained through SelectKBest
x_best - train_data with selectK_values_maintained only, data unbalanced, may contain missing values or not (val_nan makes the call)
y_best - train_data with target only, data unbalanced, may contain missing values or not (val_nan makes the call)
test_best - test_data with selectK_values_maintained only, data unbalanced, may contain missing values or not (val_nan makes the call)
missingvaluesimput - Imput a value in missing features
techuse - Run GaussianNB, SVC, SGDClassifier, RandomForestClassifier for selectK_values_maintained
otherstechuse - Run Decision Tree Classifier, AdaBoostClassifier, GaussianProcessClassifier, KNeighborsClassifier, PassiveAggressiveClassifier for selectK_values_maintained
result_best_DecisionTreeClassifier - Probability from DecisionTreeClassifier with selectK_values_maintained features, target unbalanced
result_best_AdaBoostClassifier - Probability from AdaBoostClassifier with selectK_values_maintained features, target unbalanced
result_best_GaussianProcessClassifier - Probability from GaussianProcessClassifier with selectK_values_maintained features, target unbalanced
result_best_KNeighborsClassifier - Probability from KNeighborsClassifier with selectK_values_maintained features, target unbalanced
result_best_PassiveAggressiveClassifier - Probability from PassiveAggressiveClassifier with selectK_values_maintained features, target unbalanced
catboostuse - Run CatBoostClassifier for selectK_values_maintained, target unbalanced
result_best_CatBoostClassifier - Probability from CatBoostClassifier with selectK_values_maintained features, target unbalanced
xgboostuse - Run XGBClassifier for selectK_values_maintained, target unbalanced
result_best_XGB - Probability from XGBClassifier with selectK_values_maintained features, target unbalanced
best_training - Run Support Vector Machine, Stochastic Gradient Descent, Naive Bayes and Random Forest with best features
x_train_data_balanced_pre - Data x_train_data_balanced with DataFrame format
x_best_2 - x_train_data_balanced_pre with best features
y_best_2 - same as y_train_data_balanced
test_best_2 - same as y_val
test_best_3 - x_val with best features
y_predict_GaussianNB_best - Prediction with GaussianNB, best features only and balanced data
y_true_GaussianNB_best - same as test_best_2
y_predict_SVC_best - Prediction with SVC, best features only and balanced data
y_true_SVC_best - same as test_best_2
y_predict_SGDClassifier_best - Prediction with SGDClassifier, best features only and balanced data
y_true_SGDClassifier_best - same as test_best_2
y_predict_RandomForestClassifier_best - Prediction with RandomForestClassifier, best features only and balanced data
y_true_RandomForestClassifier_best - same as test_best_2
