
## Linear Regression Notebook - MC886 - 2017 2 semestre
## Prof: Sandra Avila

## @guilherme.brunhole
## @guilherme.mazzariol


import sys
import pickle
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile,f_classif
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier

#!/usr/bin/python
import numpy as np

def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """

    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print "error: key ", feature, " not present"
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'target' class as criteria.
        if features[0] == 'target':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)


def targetFeatureSplit( data ):
    """ 
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as 
        input formats when training/predicting)
    """
    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features

def get_score (features_perc):

	selPerc = SelectPercentile(f_classif,percentile=features_perc) # Built the SelectPercentile
	selPerc.fit(scaled,train['target']) # Learn the Features, knowing which features to use

	features_percentiled = scaled.columns[selPerc.get_support()].tolist() #Filter columns based on what Percentile support
	scaled['target'] = train['target'] #rejoin the label

	features_list = ['target'] +features_percentiled # target need to be the first one

	### Extract features and labels from dataset for local testing
	data = featureFormat(train, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)

	features_train, features_test, labels_train, labels_test = \
	    train_test_split(features, labels, test_size=0.3, random_state=42)

	clf = SGDClassifier()

	clf.fit(features_train, labels_train)
	acc_lm = clf.score(features_test, labels_test)

	return acc_lm


test_file = pd.read_csv("/guibrunhole/unicamp/mc866/p1/year-prediction-msd-test.txt")
train_file = pd.read_csv("/guibrunhole/unicamp/mc866/p1/year-prediction-msd-train.txt")

print 'Total de registros no arquivo de teste: ', len(test_file)
print 'Total de registros no arquivo de treino: ', len(train_file)

new_columns_names = []

new_columns_names.append('target')

for i in range(len(train_file.columns) -1):
    name = 'column_'+str(i)
    new_columns_names.append(name)

old_columns_names = train_file.columns
train_file.rename(columns=dict(zip(old_columns_names, new_columns_names)), inplace=True)

old_columns_names = test_file.columns
test_file.rename(columns=dict(zip(old_columns_names, new_columns_names)), inplace=True)

test = test_file[new_columns_names].copy().applymap(lambda x: 0  if x == 'NaN' else x) #replace NaN features as 0
train = train_file[new_columns_names].copy().applymap(lambda x: 0  if x == 'NaN' else x) #replace NaN features as 0

cols = train.columns.tolist()
cols.remove('target')  # removing this feature for predict it

final = train[cols].copy() #replace NaN features as 0

### Scalling all features
scaled = final.apply(MinMaxScaler().fit_transform) 

## features selection
acc_50 = get_score(50)
print "Linear Model with 50% of features accuracy is: ", acc_50

acc_25 = get_score(25)
print "Linear Model with 25% of features accuracy is: ", acc_25

acc_10 = get_score(10)
print "Linear Model with 10% of features accuracy is: ", acc_10

acc_5 = get_score(5)
print "Linear Model with 25% of features accuracy is: ", acc_5