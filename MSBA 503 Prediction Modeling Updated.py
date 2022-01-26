# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:35:42 2021

@author: gidonbonner
"""

pip install xgboost

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Create file_path variable to easily reference file path for reading files
file_path = 'C:/Users/bonne/Downloads/'

# Read and Create AllData DataFrame
AllData = pd.read_pickle(file_path + 'AllData_Model_Building_Start_Data')

#Remove all columns with 85% or more null values
AllData = AllData.dropna(axis=1, thresh=int(0.85*len(AllData)))

#Remove all rows with 90% or more null values
AllData = AllData.dropna(axis=0, thresh=int(0.9*len(AllData.iloc[0,:])))

#sklean learning algorithms take two arrays as input, one array with the class (the dependent variable) and one array with all the features (the independent variables). The following code creates the sklearn input X and y arrays and change 'Fraud'/'NotFraud' to 1/0:
X = AllData.iloc[:,10:94]
y = AllData.iloc[:,94]

#First create a list of column names consisting of all columns that have non-binary data. In the code below, (X!=0) & (X!=1) & (X!=None) returns true if the value in a cell is anything except 0, 1, or None. The code then counts how many rows are true (sum method) in each column (axis=0) and if at least one rows is true (then sum > 0) then at least one value most be something different than 0, 1, or None in that specific column and the column is selected.
X = X.loc[:,((X!=0) & (X!=1) & (X!=None)).sum(axis=0)>0]
from sklearn.pipeline import Pipeline

SimplePipeline = Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', RobustScaler()),
                ('transformer', PowerTransformer()),
                ('features',  PCA(n_components=5)),
                ('classifier', SVC())
                ])

############# Cross-Validation ############# 
from sklearn.model_selection import cross_val_score, StratifiedKFold
scores = cross_val_score(SimplePipeline, X, y, cv=StratifiedKFold(3, shuffle=True), scoring="roc_auc")

#best scores --> array([0.61777211, 0.57043651, 0.64928711]) - Simpleimputer,robustscaler, PowerTransformer, PCA, SVC 

#scores --> array([0.58120748, 0.51232993, 0.57254683]) - Simpleimputer,standardscaler, PCA, SVC
#scores --> array([0.64611678, 0.54804422, 0.60134191]) - Simpleimputer,standardscaler, QuntileTransformer, PCA, SVC
#scores --> array([0.54776077, 0.58404195, 0.63516914]) - Simpleimputer,standardscaler, PowerTransformer, PCA, SVC
#scores --> array([0.59693878, 0.64781746, 0.69918926]) - Simpleimputer,robustscaler, PowerTransformer, SelectKBest(), SVC 

############# GridSearch ############# 
from sklearn.model_selection import ParameterGrid
list(ParameterGrid({'C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
                    'gamma': ['auto', 'scale'], 
                    'class_weight': ['Balanced', 'None'],
                    'kernel': ['rbf', 'poly'],
                    'probability': [True]}))

############# Combining PipeLines, GridSearch, and Cross-Validation ############# 
from sklearn.model_selection import GridSearchCV
pd.DataFrame(SimplePipeline.get_params().keys()).sort_values(by=0)

ParamGrid = {'features__n_components': [5, 10, 15],
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             }

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=10, n_jobs=4, pre_dispatch=8)

GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_   #0.6899604210024953
GridCrossValResults.best_params_  #{'classifier__C': 0.01, 'classifier__gamma': 'auto', 'classifier__kernel': 'rbf', 'features__n_components': 15}

results = pd.DataFrame(GridCrossValResults.cv_results_)

ParamGrid = {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'features': [PCA(), TruncatedSVD()],
             'transformer': [PowerTransformer()],
             'features__n_components': [5, 10, 15],
             'classifier': [SVC(), SVR()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             }

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=1, n_jobs=4, pre_dispatch=8)
GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_ # 0.6625327452137628
GridCrossValResults.best_params_ 

######## Comparing Different Pre-Processing Hyperparameter Values ########
ParamGrid = [{'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'features': [PCA(), TruncatedSVD()],
             'transformer': [PowerTransformer()],
             'features__n_components': [5, 10, 15],
             'classifier': [SVC()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             },
             {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'transformer': [PowerTransformer()],
             'features': [PCA(), TruncatedSVD()],
             'features__n_components': [5, 10, 15],
             'classifier': [LogisticRegression()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2],  
             'classifier__solver': ['saga', 'liblinear']
             },
             {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'transformer': [PowerTransformer()],
             'features': [PCA(), TruncatedSVD()],
             'features__n_components': [5, 10, 15],
             'classifier': [RandomForestClassifier()], 
             'classifier__min_samples_leaf': [50, 100, 150, 300],
             'classifier__min_samples_split': [100, 200, 300, 500]
             }]

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=1, n_jobs=4, pre_dispatch=8)
GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_ #   0.7056046733761997
GridCrossValResults.best_params_ 
#{'classifier': SVC(C=1.5, gamma='auto'),
# 'classifier__C': 1.5,
# 'classifier__gamma': 'auto',
# 'classifier__kernel': 'rbf',
# 'features': PCA(n_components=15),
# 'features__n_components': 15,
# 'imputer': SimpleImputer(),
# 'scaler': RobustScaler(),
# 'transformer': PowerTransformer()}

#2a bankruptcy financial information and the sentiment variables
#sklean learning algorithms take two arrays as input, one array with the class (the dependent variable) and one array with all the features (the independent variables). The following code creates the sklearn input X and y arrays and change 'Fraud'/'NotFraud' to 1/0:
financial = AllData.iloc[:,10:94]
sentiment_variables = AllData.iloc[:,96:104]

X = pd.concat([financial, sentiment_variables], axis = 1)
y = AllData.iloc[:,94]


#First create a list of column names consisting of all columns that have non-binary data. In the code below, (X!=0) & (X!=1) & (X!=None) returns true if the value in a cell is anything except 0, 1, or None. The code then counts how many rows are true (sum method) in each column (axis=0) and if at least one rows is true (then sum > 0) then at least one value most be something different than 0, 1, or None in that specific column and the column is selected.
X = X.loc[:,((X!=0) & (X!=1) & (X!=None)).sum(axis=0)>0]

from sklearn.pipeline import Pipeline

SimplePipeline = Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', RobustScaler()),
                ('transformer', PowerTransformer()),
                ('features',  PCA(n_components=5)),
                ('classifier', SVC())
                ])

############# Cross-Validation ############# 
from sklearn.model_selection import cross_val_score, StratifiedKFold
scores = cross_val_score(SimplePipeline, X, y, cv=StratifiedKFold(3, shuffle=True), scoring="roc_auc")

#best scores --> array([0.63973923, 0.59722222, 0.54906346]) - Simpleimputer,robustscaler, PowerTransformer, PCA, SVC 


############# GridSearch ############# 
from sklearn.model_selection import ParameterGrid
list(ParameterGrid({'C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
                    'gamma': ['auto', 'scale'], 
                    'class_weight': ['Balanced', 'None'],
                    'kernel': ['rbf', 'poly'],
                    'probability': [True]}))

############# Combining PipeLines, GridSearch, and Cross-Validation ############# 
from sklearn.model_selection import GridSearchCV
pd.DataFrame(SimplePipeline.get_params().keys()).sort_values(by=0)

ParamGrid = {'features__n_components': [5, 10, 15],
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             }

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=10, n_jobs=4, pre_dispatch=8)

GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_   #0.6752765223288707
GridCrossValResults.best_params_  #{'classifier__C': 1.5, 'classifier__gamma': 'auto', 'classifier__kernel': 'rbf', 'features__n_components': 15}

results = pd.DataFrame(GridCrossValResults.cv_results_)

ParamGrid = {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'features': [PCA(), TruncatedSVD()],
             'transformer': [PowerTransformer()],
             'features__n_components': [5, 10, 15],
             'classifier': [SVC(), SVR()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             }

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=1, n_jobs=4, pre_dispatch=8)
GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_ # 0.6979493471665682
GridCrossValResults.best_params_ 

######## Comparing Different Pre-Processing Hyperparameter Values ########
ParamGrid = [{'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'features': [PCA(), TruncatedSVD()],
             'transformer': [PowerTransformer()],
             'features__n_components': [5, 10, 15],
             'classifier': [SVC()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             },
             {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'transformer': [PowerTransformer()],
             'features': [PCA(), TruncatedSVD()],
             'features__n_components': [5, 10, 15],
             'classifier': [LogisticRegression()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2],  
             'classifier__solver': ['saga', 'liblinear']
             },
             {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'transformer': [PowerTransformer()],
             'features': [PCA(), TruncatedSVD()],
             'features__n_components': [5, 10, 15],
             'classifier': [RandomForestClassifier()], 
             'classifier__min_samples_leaf': [50, 100, 150, 300],
             'classifier__min_samples_split': [100, 200, 300, 500]
             }]

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=1, n_jobs=4, pre_dispatch=8)
GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_ #   0.7014196279729549
GridCrossValResults.best_params_ 
#{'classifier': SVC(C=0.1, gamma='auto'),
# 'classifier__C': 0.1,
# 'classifier__gamma': 'auto',
# 'classifier__kernel': 'rbf',
# 'features': PCA(n_components=15),
# 'features__n_components': 15,
# 'imputer': SimpleImputer(),
# 'scaler': StandardScaler(),
# 'transformer': PowerTransformer()}

#2b bankruptcy financial information and the sentiment variables and words not lemmatized or stemmed variables
#sklean learning algorithms take two arrays as input, one array with the class (the dependent variable) and one array with all the features (the independent variables). The following code creates the sklearn input X and y arrays and change 'Fraud'/'NotFraud' to 1/0:
financial = AllData.iloc[:,10:94]
sentiment_variables = AllData.iloc[:,96:104]
words_not_lemmatized_or_stemmed = AllData.iloc[:,104:114]

X = pd.concat([financial, sentiment_variables, words_not_lemmatized_or_stemmed], axis = 1)
y = AllData.iloc[:,94]


#First create a list of column names consisting of all columns that have non-binary data. In the code below, (X!=0) & (X!=1) & (X!=None) returns true if the value in a cell is anything except 0, 1, or None. The code then counts how many rows are true (sum method) in each column (axis=0) and if at least one rows is true (then sum > 0) then at least one value most be something different than 0, 1, or None in that specific column and the column is selected.
X = X.loc[:,((X!=0) & (X!=1) & (X!=None)).sum(axis=0)>0]

from sklearn.pipeline import Pipeline

SimplePipeline = Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', RobustScaler()),
                ('transformer', PowerTransformer()),
                ('features',  PCA(n_components=5)),
                ('classifier', SVC())
                ])

############# Cross-Validation ############# 
from sklearn.model_selection import cross_val_score, StratifiedKFold
scores = cross_val_score(SimplePipeline, X, y, cv=StratifiedKFold(3, shuffle=True), scoring="roc_auc")

#best scores --> array([0.61592971, 0.68757086, 0.57324574]) - Simpleimputer,robustscaler, PowerTransformer, PCA, SVC 


############# GridSearch ############# 
from sklearn.model_selection import ParameterGrid
list(ParameterGrid({'C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
                    'gamma': ['auto', 'scale'], 
                    'class_weight': ['Balanced', 'None'],
                    'kernel': ['rbf', 'poly'],
                    'probability': [True]}))

############# Combining PipeLines, GridSearch, and Cross-Validation ############# 
from sklearn.model_selection import GridSearchCV
pd.DataFrame(SimplePipeline.get_params().keys()).sort_values(by=0)

ParamGrid = {'features__n_components': [5, 10, 15],
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             }

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=10, n_jobs=4, pre_dispatch=8)

GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_   #0.6967780262790048
GridCrossValResults.best_params_  #{'classifier__C': 1.25, 'classifier__gamma': 'auto', 'classifier__kernel': 'rbf', 'features__n_components': 10}

results = pd.DataFrame(GridCrossValResults.cv_results_)

ParamGrid = {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'features': [PCA(), TruncatedSVD()],
             'transformer': [PowerTransformer()],
             'features__n_components': [5, 10, 15],
             'classifier': [SVC(), SVR()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             }

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=1, n_jobs=4, pre_dispatch=8)
GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_ # 0.7018153532341399
GridCrossValResults.best_params_ 

######## Comparing Different Pre-Processing Hyperparameter Values ########
ParamGrid = [{'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'features': [PCA(), TruncatedSVD()],
             'transformer': [PowerTransformer()],
             'features__n_components': [5, 10, 15],
             'classifier': [SVC()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             },
             {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'transformer': [PowerTransformer()],
             'features': [PCA(), TruncatedSVD()],
             'features__n_components': [5, 10, 15],
             'classifier': [LogisticRegression()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2],  
             'classifier__solver': ['saga', 'liblinear']
             },
             {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'transformer': [PowerTransformer()],
             'features': [PCA(), TruncatedSVD()],
             'features__n_components': [5, 10, 15],
             'classifier': [RandomForestClassifier()], 
             'classifier__min_samples_leaf': [50, 100, 150, 300],
             'classifier__min_samples_split': [100, 200, 300, 500]
             }]

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=1, n_jobs=4, pre_dispatch=8)
GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_ #   0.7016296244525208
GridCrossValResults.best_params_ 
#{'classifier': SVC(C=0.75, kernel='poly'),
# 'classifier__C': 0.75,
# 'classifier__gamma': 'scale',
# 'classifier__kernel': 'poly',
# 'features': PCA(n_components=15),
# 'features__n_components': 15,
# 'imputer': SimpleImputer(),
# 'scaler': StandardScaler(),
# 'transformer': PowerTransformer()}

#2c bankruptcy financial information and the sentiment variables and the words not lemmatized and stemmed and the words that are lemmatized but not stemmed
#sklean learning algorithms take two arrays as input, one array with the class (the dependent variable) and one array with all the features (the independent variables). The following code creates the sklearn input X and y arrays and change 'Fraud'/'NotFraud' to 1/0:
financial = AllData.iloc[:,10:94]
sentiment_variables = AllData.iloc[:,96:104]
words_not_lemmatized_or_stemmed = AllData.iloc[:,104:114]
words_lemmatized_but_not_stemmed = AllData.iloc[:,114:124]

X = pd.concat([financial, sentiment_variables, words_not_lemmatized_or_stemmed, words_lemmatized_but_not_stemmed], axis = 1)
y = AllData.iloc[:,94]


#First create a list of column names consisting of all columns that have non-binary data. In the code below, (X!=0) & (X!=1) & (X!=None) returns true if the value in a cell is anything except 0, 1, or None. The code then counts how many rows are true (sum method) in each column (axis=0) and if at least one rows is true (then sum > 0) then at least one value most be something different than 0, 1, or None in that specific column and the column is selected.
X = X.loc[:,((X!=0) & (X!=1) & (X!=None)).sum(axis=0)>0]

from sklearn.pipeline import Pipeline

SimplePipeline = Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', RobustScaler()),
                ('transformer', PowerTransformer()),
                ('features',  PCA(n_components=5)),
                ('classifier', SVC())
                ])

############# Cross-Validation ############# 
from sklearn.model_selection import cross_val_score, StratifiedKFold
scores = cross_val_score(SimplePipeline, X, y, cv=StratifiedKFold(3, shuffle=True), scoring="roc_auc")

#best scores --> array([0.55725624, 0.62896825, 0.55325692]) - Simpleimputer,robustscaler, PowerTransformer, PCA, SVC 


############# GridSearch ############# 
from sklearn.model_selection import ParameterGrid
list(ParameterGrid({'C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
                    'gamma': ['auto', 'scale'], 
                    'class_weight': ['Balanced', 'None'],
                    'kernel': ['rbf', 'poly'],
                    'probability': [True]}))

############# Combining PipeLines, GridSearch, and Cross-Validation ############# 
from sklearn.model_selection import GridSearchCV
pd.DataFrame(SimplePipeline.get_params().keys()).sort_values(by=0)

ParamGrid = {'features__n_components': [5, 10, 15],
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             }

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=10, n_jobs=4, pre_dispatch=8)

GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_   #0.6889411776887314
GridCrossValResults.best_params_  #{'classifier__C': 0.01, 'classifier__gamma': 'auto', 'classifier__kernel': 'rbf', 'features__n_components': 15}

results = pd.DataFrame(GridCrossValResults.cv_results_)

ParamGrid = {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'features': [PCA(), TruncatedSVD()],
             'transformer': [PowerTransformer()],
             'features__n_components': [5, 10, 15],
             'classifier': [SVC(), SVR()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             }

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=1, n_jobs=4, pre_dispatch=8)
GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_ # 0.7046184341316435
GridCrossValResults.best_params_ 

######## Comparing Different Pre-Processing Hyperparameter Values ########
ParamGrid = [{'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'features': [PCA(), TruncatedSVD()],
             'transformer': [PowerTransformer()],
             'features__n_components': [5, 10, 15],
             'classifier': [SVC()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             },
             {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'transformer': [PowerTransformer()],
             'features': [PCA(), TruncatedSVD()],
             'features__n_components': [5, 10, 15],
             'classifier': [LogisticRegression()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2],  
             'classifier__solver': ['saga', 'liblinear']
             },
             {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'transformer': [PowerTransformer()],
             'features': [PCA(), TruncatedSVD()],
             'features__n_components': [5, 10, 15],
             'classifier': [RandomForestClassifier()], 
             'classifier__min_samples_leaf': [50, 100, 150, 300],
             'classifier__min_samples_split': [100, 200, 300, 500]
             }]

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=1, n_jobs=4, pre_dispatch=8)
GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_ #   0.7211427950175503
GridCrossValResults.best_params_ 
# {'classifier': SVC(C=0.5, kernel='poly'),
#'classifier__C': 0.5,
# 'classifier__gamma': 'scale',
# 'classifier__kernel': 'poly',
# 'features': TruncatedSVD(n_components=15),
# 'features__n_components': 15,
# 'imputer': SimpleImputer(),
# 'scaler': StandardScaler(),
# 'transformer': PowerTransformer()}

#2d bankruptcy financial information and the sentiment variables and the words not lemmatized and stemmed and the words that are lemmatized but not stemmed and the words that are lemmatized and stemmed
#sklean learning algorithms take two arrays as input, one array with the class (the dependent variable) and one array with all the features (the independent variables). The following code creates the sklearn input X and y arrays and change 'Fraud'/'NotFraud' to 1/0:
financial = AllData.iloc[:,10:94]
sentiment_variables = AllData.iloc[:,96:104]
words_not_lemmatized_or_stemmed = AllData.iloc[:,104:114]
words_lemmatized_but_not_stemmed = AllData.iloc[:,114:124]
words_lemmatized_and_stemmed = AllData.iloc[:,124:134]

X = pd.concat([financial, sentiment_variables, words_not_lemmatized_or_stemmed, words_lemmatized_but_not_stemmed, words_lemmatized_and_stemmed], axis = 1)
y = AllData.iloc[:,94]


#First create a list of column names consisting of all columns that have non-binary data. In the code below, (X!=0) & (X!=1) & (X!=None) returns true if the value in a cell is anything except 0, 1, or None. The code then counts how many rows are true (sum method) in each column (axis=0) and if at least one rows is true (then sum > 0) then at least one value most be something different than 0, 1, or None in that specific column and the column is selected.
X = X.loc[:,((X!=0) & (X!=1) & (X!=None)).sum(axis=0)>0]

from sklearn.pipeline import Pipeline

SimplePipeline = Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', RobustScaler()),
                ('transformer', PowerTransformer()),
                ('features',  PCA(n_components=5)),
                ('classifier', SVC())
                ])

############# Cross-Validation ############# 
from sklearn.model_selection import cross_val_score, StratifiedKFold
scores = cross_val_score(SimplePipeline, X, y, cv=StratifiedKFold(3, shuffle=True), scoring="roc_auc")

#best scores --> array([0.47902494, 0.5710034 , 0.49175287]) - Simpleimputer,robustscaler, PowerTransformer, PCA, SVC 


############# GridSearch ############# 
from sklearn.model_selection import ParameterGrid
list(ParameterGrid({'C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
                    'gamma': ['auto', 'scale'], 
                    'class_weight': ['Balanced', 'None'],
                    'kernel': ['rbf', 'poly'],
                    'probability': [True]}))

############# Combining PipeLines, GridSearch, and Cross-Validation ############# 
from sklearn.model_selection import GridSearchCV
pd.DataFrame(SimplePipeline.get_params().keys()).sort_values(by=0)

ParamGrid = {'features__n_components': [5, 10, 15],
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             }

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=10, n_jobs=4, pre_dispatch=8)

GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_   #0.7109529504343595
GridCrossValResults.best_params_  #{'classifier__C': 0.01, 'classifier__gamma': 'auto', 'classifier__kernel': 'rbf', 'features__n_components': 15}

results = pd.DataFrame(GridCrossValResults.cv_results_)

ParamGrid = {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'features': [PCA(), TruncatedSVD()],
             'transformer': [PowerTransformer()],
             'features__n_components': [5, 10, 15],
             'classifier': [SVC(), SVR()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             }

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=1, n_jobs=4, pre_dispatch=8)
GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_ # 0.6917500828337424
GridCrossValResults.best_params_ 

######## Comparing Different Pre-Processing Hyperparameter Values ########
ParamGrid = [{'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'features': [PCA(), TruncatedSVD()],
             'transformer': [PowerTransformer()],
             'features__n_components': [5, 10, 15],
             'classifier': [SVC()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], 
             'classifier__gamma': ['auto', 'scale'], 
             'classifier__kernel': ['rbf', 'poly']
             },
             {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'transformer': [PowerTransformer()],
             'features': [PCA(), TruncatedSVD()],
             'features__n_components': [5, 10, 15],
             'classifier': [LogisticRegression()], 
             'classifier__C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2],  
             'classifier__solver': ['saga', 'liblinear']
             },
             {'imputer': [SimpleImputer()],
             'scaler': [StandardScaler(), RobustScaler()],
             'transformer': [PowerTransformer()],
             'features': [PCA(), TruncatedSVD()],
             'features__n_components': [5, 10, 15],
             'classifier': [RandomForestClassifier()], 
             'classifier__min_samples_leaf': [50, 100, 150, 300],
             'classifier__min_samples_split': [100, 200, 300, 500]
             }]

GridCrossValResults = GridSearchCV(SimplePipeline, param_grid=ParamGrid, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True), verbose=1, n_jobs=4, pre_dispatch=8)
GridCrossValResults.fit(X, y)
GridCrossValResults.best_score_ #   0.6788386968181488
GridCrossValResults.best_params_ 
# {'classifier': SVC(C=1, gamma='auto'),
# 'classifier__C': 1,
# 'classifier__gamma': 'auto',
# 'classifier__kernel': 'rbf',
# 'features': TruncatedSVD(n_components=15),
# 'features__n_components': 15,
# 'imputer': SimpleImputer(),
# 'scaler': RobustScaler(),
# 'transformer': PowerTransformer()}