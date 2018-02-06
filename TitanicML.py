import sys, os, re, csv, codecs, cPickle, numpy as np, pandas as pd
import xgboost as xgb
from sklearn import decomposition,neighbors
from sklearn.neighbors import KDTree,BallTree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.cross_validation import KFold
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

def shuffle(to_shuffle):
    random_index = np.arange(len(to_shuffle))
    np.random.shuffle(random_index)
    return to_shuffle[random_index]

from set_params import *


def fix_null(df, col):
    for title in np.unique(df.Title[df[col].isnull()]):
        for sex in np.unique(df.Sex[(df[col].isnull()) & (df.Title == title)]):
            for Pclass in np.unique(df.Pclass[(df[col].isnull()) & (df.Title == title) & (df.Sex == sex)]):
                df[col][(df[col].isnull()) & (df.Title == title) & (df.Sex == sex) & (df.Pclass == Pclass)] = np.average(df[col][(df[col].isnull() == False) & (df.Title == title) & (df.Sex == sex) & (df.Pclass == Pclass)])
    return df

def setup_csv(filename, returnid=False, onehotencoding=True, drop_first_one_hot=True, use_isalone=False, normalizefare=False, cabinletters=False, tier=False):
    initial_df = pd.read_csv(filename)
    initial_df['Title'] = np.array([str.split('.')[0].split(' ')[-1] for str in initial_df.Name.values])
    
    try:
        df = initial_df[['Survived', 'Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin']]
    except:
        df = initial_df[['Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin']]

    for Pclass in np.unique(df.Pclass):
        df.Fare[(df.Fare.isnull()) & (df.Pclass == Pclass)] = np.average(df.Fare[(df.Fare.isnull() == False) & (df.Pclass == Pclass)])

    if normalizefare:
        df['normalizefare'] = df.groupby('Pclass')['Fare'].transform(lambda x: (x - x.mean())/x.std())
        
        if tier:
            df['tier'] = 0
            for Pclass in np.unique(df.Pclass):
                quantiles = df[df.Pclass == Pclass]['normalizefare'].quantile([0.25, 0.5, 0.75])
                df.tier[df.Pclass == Pclass] = Pclass + np.searchsorted(quantiles, df[df.Pclass == Pclass]['normalizefare'])
                

    if cabinletters:
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        for letter in letters: 
            df['Cabin_{}'.format(letter)] = False
        for i in range(0, np.shape(df)[0]):
            for letter in letters: 
                try:
                    df['Cabin_{}'.format(letter)].iloc[i] = letter in df.Cabin.iloc[i]
                except TypeError:
                    pass

    df.Cabin = pd.notnull(df.Cabin)
            

    df.Embarked[df.Embarked.isnull()] = df.Embarked.mode()[0]

    df.Title[np.in1d(df.Title, ['Mlle', 'Ms'])] = 'Miss'
    df.Title[df.Title == 'Mme'] = 'Mrs'
    df.Title[np.in1d(df.Title, ['Mr', 'Miss', 'Mrs', 'Master'], invert = True)] = 'Other'

    df = fix_null(df, 'Age')
    needencoding = ['Pclass', 'Embarked', 'SibSp', 'Parch', 'Title']
    if use_isalone:
        df['family_size'] = df.SibSp + df.Parch
        df['not_alone'] = np.array(df.SibSp + df.Parch, dtype = 'bool')
        #df = df.loc[:, np.in1d(df.columns, ['SibSp', 'Parch'])]
        #needencoding = ['Pclass', 'Embarked', 'Title']

    #df.SibSp = np.array(df.SibSp.values, dtype = 'bool')
    #df.Parch = np.array(df.Parch.values, dtype = 'bool')
    df.SibSp[df.SibSp > 1] = 2
    df.Parch[df.Parch > 1] = 2



    #onehotencoding
    if onehotencoding:
        df.Sex = df.Sex == 'female'
        for to_encode in needencoding:
            df = pd.merge(df.loc[:, df.columns != to_encode], pd.get_dummies(df[to_encode], prefix=to_encode, drop_first=drop_first_one_hot), left_index=True, right_index=True)

    if returnid:
        return df, initial_df['PassengerId']
    else:
        return df

def RF(params):
    NE = params[0]
    clf = RandomForestClassifier(n_estimators = NE)
    return clf

def logreg(params):
    clf = LogisticRegression()
    return clf

def pred(datadir, classifier=RF, clfparams=[10], nfolds=5, use_isalone=False, normalizefare=False, cabinletters=False, tier=False, onehotencoding=True, drop_first_one_hot=True):
    testfile = '{}/test.csv'.format(datadir)
    trainfile = '{}/train.csv'.format(datadir)
    testdf, testid = setup_csv(testfile, True, use_isalone=use_isalone, normalizefare=normalizefare, cabinletters=cabinletters, tier=tier, onehotencoding=onehotencoding, drop_first_one_hot=drop_first_one_hot)
    traindf = setup_csv(trainfile, use_isalone=use_isalone, normalizefare=normalizefare, cabinletters=cabinletters, tier=tier, onehotencoding=onehotencoding, drop_first_one_hot=drop_first_one_hot)
    Xtrain = traindf.loc[:, traindf.columns != 'Survived']
    Xtest = testdf
    ytrain = traindf['Survived']
    try:
        clf = SklearnHelper(clf=classifier, params=clf_params)
    except:
        clf = classifier(clfparams)
    clf.fit(Xtrain, ytrain)
    predictions = clf.predict(Xtest)
    cross_val_scores = cross_val_score(clf, Xtrain, ytrain, cv=nfolds)
    return predictions, cross_val_scores

class SklearnHelper(object):
    def __init__(self, clf, seed=None, params={}):
        params_to_set = dict(params)
        if seed != None: params_to_set['random_state'] = seed
        self.clf = clf(**params_to_set)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self, x, y, verbose=True):
        importances = self.clf.fit(x,y).feature_importances_
        if verbose:
            colnames = x.columns.tolist()
            isrt = np.argsort(importances)
            for i in isrt: 
                print('{}: {:.6f}'.format(colnames[i], importances[i]))
        print(importances)

    def Kfold(self, x_train, y_train, nfolds=5):
        return cross_val_score(self.clf, x_train, y_train, cv=nfolds)

def get_oof(clf, x_train, y_train, x_test, nfolds=5):
    ntrain, ntest = x_train.shape[0], x_test.shape[0]
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_pre_average = np.empty((nfolds, ntest))
    shuffle_inds = np.arange(ntrain)
    np.random.shuffle(shuffle_inds)

    for ikfold in range(0, nfolds):
        test_index, train_index = shuffle_inds[int(ikfold*ntrain/nfolds):int((ikfold+1)*ntrain/nfolds)], np.append(shuffle_inds[:int(ikfold*ntrain/nfolds)], shuffle_inds[int((ikfold+1)*ntrain/nfolds):])
        x_train_curfold = x_train_curfoldain.iloc[train_index]
        y_train_curfold = y_train_curfoldain.iloc[train_index]
        x_test_curfold = x_train_curfoldain.iloc[test_index]

        clf.train(x_train_curfold, y_train_curfold)

        oof_train[test_index] = clf.predict(x_test_curfold)
        oof_test_pre_average[ikfold, :] = clf.predict(x_test_curfoldst)

    oof_test[:] = oof_test_pre_average.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

clf_long_strings, clf_short_strings = np.array(['RandomForestClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'SVC', 'LogisticRegression']), np.array(['rf', 'et', 'ada', 'gb', 'svc', 'lr'])
#clf_long_strings, clf_short_strings = np.array(['RandomForestClassifier', 'ExtraTreesClassifier']), np.array(['rf', 'et'])

for clf_LS, clf_SS in zip(clf_long_strings, clf_short_strings):
    exec('{} = SklearnHelper(clf={}, params={}_params)'.format(clf_SS, clf_LS, clf_SS))

def stack_pred(datadir='.', params_dict=None, use_isalone=False, checkpreds=True, train_frac=0.7, normalizefare=False, cabinletters=False, tier=False, onehotencoding=True, drop_first_one_hot=True, feature_slice=None):
    testfile, trainfile = '{}/test.csv'.format(datadir), '{}/train.csv'.format(datadir)
    testdf, testid = setup_csv(testfile, True, use_isalone=use_isalone, normalizefare=normalizefare, cabinletters=cabinletters, tier=tier, onehotencoding=onehotencoding, drop_first_one_hot=drop_first_one_hot)
    traindf = setup_csv(trainfile, use_isalone=use_isalone, normalizefare=normalizefare, cabinletters=cabinletters, tier=tier, onehotencoding=onehotencoding, drop_first_one_hot=drop_first_one_hot)
    if checkpreds:
        ntrain = traindf.shape[0]
        randinds = shuffle(np.arange(ntrain))
        traindf = traindf.iloc[randinds[:int(ntrain*train_frac)]]
        testdf =  traindf.iloc[randinds[int(ntrain*train_frac):]]
        ytest = testdf['Survived']
    Xtrain = traindf.loc[:, traindf.columns != 'Survived']
    Xtest = testdf.loc[:, testdf.columns != 'Survived']
    if feature_slice != None:
        Xtrain = Xtrain[feature_slice]
        Xtest =  Xtest[feature_slice]
    ytrain = traindf['Survived']
    ntrain = Xtrain.shape[0] 
    ntest =  Xtest.shape[0]

    Xsecondary_train = np.zeros((ntrain, 0))
    Xsecondary_test = np.zeros((ntest, 0))
    for clf_LS, clf_SS in zip(clf_long_strings, clf_short_strings):
        if params_dict != None:
            exec("{} = SklearnHelper(clf={}, params=params_dict['{}'])".format(clf_SS, clf_LS, clf_SS))
        else:
            exec('{} = SklearnHelper(clf={}, params={}_params)'.format(clf_SS, clf_LS, clf_SS))
        exec('{}_train, {}_test = get_oof({}, Xtrain, ytrain, Xtest)'.format(clf_SS, clf_SS, clf_SS))
        exec('Xsecondary_train = np.concatenate((Xsecondary_train, {}_train), axis = 1)'.format(clf_SS))
        exec('Xsecondary_test = np.concatenate((Xsecondary_test, {}_test), axis = 1)'.format(clf_SS))
        if checkpreds: 
            exec('preds = {}.predict(Xtest)'.format(clf_SS))
            accuracy = sum(preds == ytest) * 1./ ntest
            print '{} Trained. Accuracy: {}%'.format(clf_LS, accuracy * 100)
        else:
            print '{} Trained'.format(clf_LS)
    gbm = SklearnHelper(clf=xgb.XGBClassifier, params=gbm_params)
    gbm.fit(Xsecondary_train, ytrain)
    print 'XGBoost Trained'
    predictions = gbm.predict(Xsecondary_test)
    if checkpreds:
        return predictions, ytest
    else:
        return predictions

def create_cv_split(n_splits=10, test_size=.3, train_size=.6):
    return model_selection.ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size)

def tune_params(clf_long_strings, clf_short_strings, Xtrain, ytrain, cv_split=None, outfile=None):
    if cv_split == None:
        cv_split = create_cv_split()
    tuned_params_dict = {'inputs': {}}
    for clf_SS, clf_LS in zip(clf_short_strings, clf_long_strings):
        exec("tuned_params_dict['inputs']['{}'] = {}_param_grid".format(clf_SS,clf_SS))
        exec("{}_tuned_model = model_selection.GridSearchCV({}(), param_grid={}_param_grid, scoring = 'roc_auc', cv = cv_split)".format(clf_SS,clf_LS,clf_SS))
        exec('{}_tuned_model.fit(Xtrain,ytrain)'.format(clf_SS))
        exec("tuned_params_dict['{}'] = {}_tuned_model.best_params_".format(clf_SS,clf_SS))
    if outfile != None: 
        cPickle.dump(tuned_params_dict, open(outfile, 'wb'))
    return tuned_params_dict

def load_tuned_params(paramfile, set_nestimators=None):
    tuned_params_dict = cPickle.load(open(paramfile, 'rb'))
    if set_nestimators != None:
        for key in tuned_params_dict.keys(): 
            try:
                tuned_params_dict[key]['n_estimators'] # Only want to do this where n_estimators is already set
                tuned_params_dict[key]['n_estimators'] = set_nestimators
            except KeyError:
                pass
    return tuned_params_dict

def plot_learning_curve(datadir, train_sizes, classifier=RF, clfparams=[10], nfolds=5, use_isalone=False, normalizefare=False, cabinletters=False, tier=False, onehotencoding=True, drop_first_one_hot=True, setup_csv=True):
    trainfile = '{}/train.csv'.format(datadir)
    if setup_csv:
        traindf = setup_csv(trainfile, use_isalone=use_isalone, normalizefare=normalizefare, cabinletters=cabinletters, tier=tier, onehotencoding=onehotencoding, drop_first_one_hot=drop_first_one_hot)
    else:
        traindf = pd.read_csv(trainfile)
    Xtrain = traindf.loc[:, traindf.columns != 'Survived']
    ytrain = traindf['Survived']
    try:
        clf = SklearnHelper(clf=classifier, params=clfparams)
        train_sizes, train_scores, test_scores = learning_curve(clf.clf, Xtrain, ytrain, cv=nfolds, train_sizes=train_sizes)
    except:
        clf = classifier(clfparams)
        train_sizes, train_scores, test_scores = learning_curve(clf, Xtrain, ytrain, cv=nfolds, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure()
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
