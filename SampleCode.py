#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import mean_squared_error



################################### Import the data, inspect and process the features and labels #################################
data = pd.read_csv("insurance_claims.csv")
data.head()
data.columns
data.isnull().any()
data["fraud_reported"] = np.where(data["fraud_reported"] == "Y", 1, 0)

# Show the ratio of normal claims V.S. fraud claims
data["fraud_reported"].describe()
f, ax = plt.subplots(figsize = (10, 10))
sns.countplot(x = 'fraud_reported', data = data)

# Check and drop useless or incomplete features
data["collision_type"].unique()
data["property_damage"].unique()
data["police_report_available"].unique()

data.drop(["policy_number", "insured_zip"], axis = 1, inplace = True)
data.drop("incident_location", axis = 1, inplace = True)
data.drop("policy_csl", axis = 1, inplace = True)
data.drop(["collision_type", "property_damage", "police_report_available"], axis = 1, inplace = True)
data.drop(["policy_bind_date", "incident_date"], axis = 1, inplace = True)

# Encode the string data type
le = LabelEncoder()
data["policy_state"] = le.fit_transform(data["policy_state"])
data["insured_sex"] = le.fit_transform(data["insured_sex"])
data["insured_education_level"] = le.fit_transform(data["insured_education_level"])
data["insured_occupation"] = le.fit_transform(data["insured_occupation"])
data["insured_hobbies"] = le.fit_transform(data["insured_hobbies"])
data["insured_relationship"] = le.fit_transform(data["insured_relationship"])
data["incident_type"] = le.fit_transform(data["incident_type"])
data["incident_severity"] = le.fit_transform(data["incident_severity"])
data["authorities_contacted"] = le.fit_transform(data["authorities_contacted"])
data["incident_state"] = le.fit_transform(data["incident_state"])
data["incident_city"] = le.fit_transform(data["incident_city"])
data["auto_make"] = le.fit_transform(data["auto_make"])
data["auto_model"] = le.fit_transform(data["auto_model"])
data.describe()

# Show the correlation between features and labels
plt.figure(figsize = (32,32))
sns.heatmap(data.corr(), annot = True, square = True, linewidths = .5, cbar_kws = {"shrink": .5})



######################################## Split the data into training set and testing set ########################################
y = data['fraud_reported']
X = data.drop('fraud_reported', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)



############################################### Normalize the features############################################### 
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Show the ratio of normal claims V.S. fraud claims in the training set and testing set
y_train.describe()
y_test.describe()



####################################### SMOTE sampling for training set due to imbalanced labels #######################################
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Show the ratio of normal claims V.S. fraud claims in the SMOTE training set
y_train_smote.describe()



############################################## Model 1: Logistict Regression ##############################################
log_reg = LogisticRegression()
# 5-Fold CV accuracy for Logistict Regression using original training set
scores = np.average(cross_val_score(log_reg, X_train, y_train, cv = 5)) 
print("CV accuracy of Logistic model using original training set: ", scores)
# 5-Fold CV accuracy for Logistict Regression using SMOTE training set
scores_smote = np.average(cross_val_score(log_reg, X_train_smote, y_train_smote, cv = 5))
print("CV accuracy of Logistic model using SMOTE training set: ", scores_smote)  



############################################## Model 1.1: Logistict LASSO Regression ##############################################
log_la_reg = LogisticRegression(penalty = "l1", solver = 'liblinear')
# 5-Fold CV accuracy for Logistict LASSO Regression using original training set
scores_la = np.average(cross_val_score(log_la_reg, X_train, y_train, cv = 5))
print("CV accuracy of Logistic LASSO model using original training set: ", scores_la)
# 5-Fold CV accuracy for Logistict LASSO Regression using SMOTE training set
scores_la_smote = np.average(cross_val_score(log_la_reg, X_train_smote, y_train_smote, cv = 5))
print("CV accuracy of Logistic LASSO model using SMOTE training set: ", scores_la_smote) 




###################################################### Model 2: KNN ###################################################### 
# Randomly split the original training set by 70% V.S. 30% into training set and validation set
maxIndex = len(X_train)
nTrain = int(len(X_train) * 0.7)
train_indices = np.random.choice(maxIndex, nTrain, replace = False)
# Training set
X_train_KNN = X_train[train_indices]    
y_train_KNN = np.array(y_train)[train_indices]
# Validation set
allind = np.array(range(len(X_train)), dtype = int)
valid_indices = np.setdiff1d(allind, train_indices)
X_valid_KNN = X_train[valid_indices]     
y_valid_KNN = np.array(y_train)[valid_indices]

# Show the validation accuracy using different K and original training set
# Choose the value of K with highest validation accuracy
scores = []
for k in range(5, 21, 1):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn_fit = knn.fit(X_train_KNN, y_train_KNN)
    scores.append(knn.score(X_valid_KNN, y_valid_KNN))
results = pd.DataFrame(columns = ["K", "knn_score"])
results["K"] = np.arange(5,21,1)
results["knn_score"] =scores
plt.scatter(results["K"], results["knn_score"])
plt.xticks(range(5, 21, 1))
plt.xlabel("Parameter K")
plt.ylabel("KNN Score")
plt.title("KNN Score for Different K Parameter using original set")
plt.grid()

# Randomly choose 70% of the SMOTE training set into training set
maxIndex = len(X_train_smote)
nTrain = int(len(X_train_smote) * 0.7)
train_indices = np.random.choice(maxIndex, nTrain, replace = False)
# SMOTE training set
X_train_smote_KNN = X_train_smote[train_indices]    
y_train_smote_KNN = np.array(y_train_smote)[train_indices]

# Show the validation accuracy using different K and SMOTE training set
# Choose the value of K with highest validation accuracy
scores = []
for k in range(5,21,1):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn_fit = knn.fit(X_train_smote_KNN, y_train_smote_KNN)
    scores.append(knn.score(X_valid_KNN, y_valid_KNN))
results_smote = pd.DataFrame(columns=["K", "knn_score"])
results_smote["K"] = np.arange(5, 21, 1)
results_smote["knn_score"] = scores
plt.scatter(results_smote["K"], results_smote["knn_score"])
plt.xticks(range(5, 21, 1))
plt.xlabel("Parameter K")
plt.ylabel("KNN Score")
plt.title("KNN Score for Different K Parameter using SMOTE set")
plt.grid()




######################################################  Model 3: Decision Tree ###################################################### 
# Choose the best combination of hyper-parameters for Decision Tree using the original training set
params = {"criterion":("gini", "entropy"),
          "splitter":("best", "random"),
          "max_depth":(5, 10, 15),
          "min_samples_split":(2, 3, 4)} # Specify hyper-parameters
clf = tree.DecisionTreeClassifier(random_state = 42)
# Find the best combination with highest 3-Fold CV accuracy
clf_cv = GridSearchCV(clf, params, scoring = "accuracy", n_jobs = -1,verbose = 1,cv = 3) 
clf_cv.fit(X_train, y_train)
best_params = clf_cv.best_params_
print("Best parameters for Decision Tree model using original set: ", best_params)

# Draw the Decision Tree
clf = tree.DecisionTreeClassifier(**best_params)
clf.fit(X_train, y_train)
plt.figure(figsize = (32, 16))
tree.plot_tree(clf)


# Choose the best combination of hyper-parameters for Decision Tree using the SMOTE training set
clf_smote = tree.DecisionTreeClassifier(random_state = 42)
# Find the best combination with highest 3-Fold CV accuracy
clf_smote_cv = GridSearchCV(clf_smote, params, scoring = "accuracy", n_jobs = -1, verbose = 1, cv = 3)
clf_smote_cv.fit(X_train_smote, y_train_smote)
best_params_smote = clf_smote_cv.best_params_
print("Best parameters for Decision Tree model using SMOTE set: ", best_params_smote)

# Draw the Decision Tree
clf_smote = tree.DecisionTreeClassifier(**best_params_smote)
clf_smote.fit(X_train_smote, y_train_smote)
plt.figure(figsize = (32, 16))
tree.plot_tree(clf_smote)




##################################################  Model 4: Support Vector Machine ##################################################
# Choose the best combination of hyper-parameters for SVM using the original training set
param_grid = {"C": [10, 100, 1000],  
              "gamma": [0.1, 0.01, 0.001], 
              "kernel": ["rbf", "sigmoid"]} # Specify hyper-parameters
# Find the best combination with highest 5-Fold CV accuracy
svm_cv = GridSearchCV(SVC(), param_grid, cv = 5, scoring = "accuracy")
svm_cv.fit(X_train, y_train)
best_params = svm_cv.best_params_
best_score = svm_cv.best_score_
print("Best parameters for Support Vector Machine using original set: ", best_params)
print("Best accuracy for Support Vector Machine using original set: ", best_score)

# Choose the best combination of hyper-parameters for SVM using the SMOTE training set
# Find the best combination with highest 5-Fold CV accuracy
svm_smote_cv = GridSearchCV(SVC(), param_grid, cv = 5, scoring = "accuracy")
svm_smote_cv.fit(X_train_smote, y_train_smote)
best_params_smote = svm_smote_cv.best_params_
best_score_smote = svm_smote_cv.best_score_
print("Best parameters for Support Vector Machine using using SMOTE set: ", best_params_smote)
print("Best accuracy for Support Vector Machine using using SMOTE set: ", best_score_smote)




################################################## Model Evaluation ##################################################



################################### Model 1: Logistict Regression ##########################################

# Model evaluation for logistic model trained by the original training set
lr = LogisticRegression().fit(X_train, y_train)
y_score = lr.predict_proba(X_test)[:, 1]
# ROC and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Logistic model trained by original training set')
plt.legend(loc="lower right")
plt.show()
# Accuracy and RMSE
pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Test accuracy for Logistic model trained by original training set: ", accuracy)
print("Test RMSE for Logistic model trained by original training set: ", rmse)


# Model evaluation for logistic model trained by the SMOTE training set
lr_smote = LogisticRegression().fit(X_train_smote, y_train_smote)
y_score = lr_smote.predict_proba(X_test)[:,1]
# ROC and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Logistic model trained by SMOTE training set')
plt.legend(loc="lower right")
plt.show()
# Accuracy and RMSE
pred = lr_smote.predict(X_test)
accuracy = accuracy_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Test accuracy for Logistic model trained by SMOTE training set: ", accuracy)
print("Test RMSE for Logistic model trained by SMOTE training set: ", rmse)




################################### Model 1.1: Logistict LASSO Regression ##########################################

# Model evaluation for logistic LASSO model trained by the original training set
lr_la = LogisticRegression(penalty = "l1",solver = 'liblinear').fit(X_train, y_train)
y_score = lr_la.predict_proba(X_test)[:,1]
# ROC and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Logistic LASSO model trained by original training set')
plt.legend(loc="lower right")
plt.show()
# Accuracy and RMSE
pred = lr_la.predict(X_test)
accuracy = accuracy_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Test accuracy for Logistic LASSO model trained by original training set: ", accuracy)
print("Test RMSE for Logistic LASSO model trained by original training set: ", rmse)


# Model evaluation for logistic LASSO model trained by the SMOTE training set
lr_la_smote = LogisticRegression(penalty = "l1",solver = 'liblinear').fit(X_train_smote, y_train_smote)
y_score = lr_la_smote.predict_proba(X_test)[:,1]
# ROC and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Logistic LASSO model trained by SMOTE training set')
plt.legend(loc="lower right")
plt.show()
# Accuracy and RMSE
pred = lr_la_smote.predict(X_test)
accuracy = accuracy_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Test accuracy for Logistic LASSO model trained by SMOTE training set: ", accuracy)
print("Test RMSE for Logistic LASSO model trained by SMOTE training set: ", rmse)




################################### Model 2: KNN ##########################################

# Model evaluation for KNN model trained by the original training set
# From the Model Construction part, we choose K=5 for both original training set and SMOTE training set
knn = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)
y_score = knn.predict_proba(X_test)[:, 1]
# ROC and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for KNN model (K=5) trained by original training set')
plt.legend(loc="lower right")
plt.show()
# Accuracy and RMSE
pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Test accuracy for KNN model (K=5) trained by original training set: ", accuracy)
print("Test RMSE for KNN model (K=5) trained by original training set: ", rmse)


# Model evaluation for KNN model trained by the SMOTE training set
knn_smote = KNeighborsClassifier(n_neighbors = 5).fit(X_train_smote, y_train_smote)
y_score = knn_smote.predict_proba(X_test)[:, 1]
# ROC and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for KNN model (K=5) trained by SMOTE training set')
plt.legend(loc="lower right")
plt.show()
# Accuracy and RMSE
pred = knn_smote.predict(X_test)
accuracy = accuracy_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Test accuracy for KNN model (K=5) trained by SMOTE training set: ", accuracy)
print("Test RMSE for KNN model (K=5) trained by SMOTE training set: ", rmse)




################################### Model 3: Decision Tree ##########################################

# Model evaluation for decision tree model trained by the original training set
# From the Model Construction part, we choose the best parameters (selection from GridSearchCV) for the original training set
clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 5, min_samples_split = 2, splitter = "best").fit(X_train, y_train)
y_score = clf.predict_proba(X_test)[:, 1]
# ROC and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Decision Tree model trained by original training set')
plt.legend(loc="lower right")
plt.show()
# Accuracy and RMSE
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Test accuracy for Decision Tree model trained by original training set: ", accuracy)
print("Test RMSE for Decision Tree model trained by original training set: ", rmse)


# Model evaluation for decision tree model trained by the SMOTE training set
# From the Model Construction part, we choose the best parameters (selection from GridSearchCV) for the SMOTE training set
clf_smote = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 5, 
                                  min_samples_split = 2, 
                                  splitter = "best").fit(X_train_smote, y_train_smote)
y_score = clf_smote.predict_proba(X_test)[:, 1]
# ROC and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Decision Tree model trained by SMOTE training set')
plt.legend(loc="lower right")
plt.show()
# Accuracy and RMSE
pred = clf_smote.predict(X_test)
accuracy = accuracy_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Test accuracy for Decision Tree model trained by SMOTE training set: ", accuracy)
print("Test RMSE for Decision Tree model trained by SMOTE training set: ", rmse)




################################### Model 4: Model 4: Support Vector Machine ##########################################

# Model evaluation for SVM trained by the original training set
# From the Model Construction part, we choose the best parameters (selection from GridSearchCV) for the original training set
svm = SVC(kernel = 'rbf', C = 100, gamma = 0.001).fit(X_train, y_train)
y_score = svm.decision_function(X_test)
# ROC and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Support Vector Machine trained by original training set')
plt.legend(loc="lower right")
plt.show()
# Accuracy and RMSE
pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Test accuracy for Support Vector Machine trained by original training set: ", accuracy)
print("Test RMSE for Support Vector Machine trained by original training set: ", rmse)


# Model evaluation for SVM trained by the SMOTE training set
# From the Model Construction part, we choose the best parameters (selection from GridSearchCV) for the SMOTE training set
svm_smote = SVC(kernel = 'rbf', C = 10, gamma = 0.1).fit(X_train_smote, y_train_smote)
y_score = svm_smote.decision_function(X_test)
# ROC and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Support Vector Machine trained by SMOTE training set')
plt.legend(loc="lower right")
plt.show()
# Accuracy and RMSE
pred = svm_smote.predict(X_test)
accuracy = accuracy_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Test accuracy for Support Vector Machine trained by SMOTE training set: ", accuracy)
print("Test RMSE for Support Vector Machine trained by SMOTE training set: ", rmse)



################################################################## END ################################################################## 

