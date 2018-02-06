rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

rf_param_grid = {'criterion': ['gini', 'entropy'],
    'n_jobs': [-1],
                 'n_estimators': [10],
 'max_depth': [2, 4, 6, 8, 10, None],
    'min_samples_leaf': [1, 2, 4, 10],
     'warm_start': [True, False], 
    'max_features' : ['log2', 'sqrt'],
 'random_state': [0]}
    

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

et_param_grid = {
    #'criterion' : ['mse', 'mae'],
    'n_jobs': [-1],
    'n_estimators':[10],
    'max_features': ['sqrt','log2'],
    'max_depth': [2, 4, 6, 8, 10, None],
    'min_samples_leaf': [1, 2, 4, 10],
    'verbose': [0]
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}
ada_param_grid = {
    'n_estimators': [100],
    'learning_rate' : [0.5, 0.75, 0.85, 0.9, 0.95, 1]
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

gb_param_grid = {
    'learning_rate' : [0.1, 0.5, 0.75],
    'n_estimators': [10],
     'max_features': ['sqrt', 'log2'],
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 4, 10],
    'warm_start': [True, False]
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

svc_param_grid = {
    'kernel' : ['linear', 'rbf'],
    'C' : [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
    }

#Logistic Regression Parameters
lr_params = {
    'solver': 'liblinear',
    'C' : 0.25
    }

lr_param_grid = {
    'n_jobs': [-1],
    'solver': ['liblinear', 'newton-cg'],
    'C' : [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
    'fit_intercept' : [True, False],
    'warm_start' : [True, False]
    }

#XGBoost Parameters
gbm_params = {
    #'learning_rate' : 0.02,
 'n_estimators': 2000,
 'max_depth': 4,
 'min_child_weight': 2,
 #'gamma':1,
 'gamma':0.9,                        
 'subsample':0.8,
 'colsample_bytree':0.8,
 'objective': 'binary:logistic',
 'nthread': -1,
 'scale_pos_weight':1}
