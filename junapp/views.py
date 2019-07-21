from django.shortcuts import render, redirect
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

# import seaborn as sns


# Create your views here.
from junapp.form import InputForm
from junapp.models import Runoff, Raindata


def rainwaterharvest(request):
    return render(request,template_name='junapp/input.html')

def checkharvesting(request):
    runoff=Runoff.objects.filter(id=request.POST.get('type'))
    for i in runoff:
        coefficient=i.coefficient
    months=Raindata.objects.all()
    totalwater=0
    monthly=[]
    max=0
    maxexpense=0
    totalsave=0
    for i in months:
        # Q=kRA
        rainwater=coefficient*i.evaporation*i.rainfall*float(request.POST.get('area'))
        if (max<rainwater):
            max=rainwater
        usage=int(request.POST.get('use'))*i.days
        waterrequired=usage-rainwater
        # calculating average water bill after rain harvesting
        if (waterrequired<500):
            monthlyprice=300
        elif (waterrequired<1000):
            monthlyprice=300+(1*(waterrequired-500))
        else:
            monthlyprice=300+(1*(500)+2*(waterrequired-1000))

        if (maxexpense<monthlyprice):
            maxexpense=monthlyprice

        # calculating average water bill before rain harvesting
        if (usage < 500):
            requiredmonthlyprice = 300
        elif (usage < 1000):
            requiredmonthlyprice = 300 + (1 * (usage - 500))
        else:
            requiredmonthlyprice = 300 + (1 * (500) + 2 * (usage - 1000))

        # total money saved per month
        totalsave=totalsave+(requiredmonthlyprice-monthlyprice)

        if (maxexpense<usage):
            maxexpense=usage

        monthly.append({'month':i.month,'rainfall':round(rainwater,2),'usage':usage,'monthlyprice':monthlyprice, 'requiredmonthlyprice':requiredmonthlyprice})
        # monthly[i.month]=round(rainwater,2)
        totalwater=totalwater+rainwater

    result={
        'totalwater':round(totalwater,2),
        'monthly':monthly,
        'max':max,
        'maxexpense':maxexpense,
        'totalsave':round(totalsave,2)
    }

    print(monthly)
    return render(request,'junapp/result.html',{'result':result})


def inputTest(request):
    return render(request, 'junapp/inputField.html')

def logi(request):
    inp = request.FILES['testinput'].name
    print(inp)
    #
    # train_df = pd.read_csv("junapp/static/junapp/data/train.csv")
    #
    # # Read CSV test data file into DataFrame
    # test_df = pd.read_csv("junapp/static/junapp/data/" + inp)
    # print('The number of samples into the train data is {}.'.format(train_df.shape[0]))
    # a = train_df.isnull().sum()
    #
    # train_data = train_df.copy()
    # train_data["How many days, immunization service is provided?"].fillna(
    #     train_df["How many days, immunization service is provided?"].median(skipna=True), inplace=True)
    # train_data["How many bed are available in this hospital?"].fillna(
    #     train_df["How many bed are available in this hospital?"].median(skipna=True), inplace=True)
    #
    # train_data.isnull().sum()
    #
    # test_data = test_df.copy()
    # test_data["How many days, immunization service is provided?"].fillna(
    #     test_df["How many days, immunization service is provided?"].median(skipna=True), inplace=True)
    #
    # a = test_data.isnull().sum()
    # print(a)
    # cols = ["Does this health facility have its own building?", "Infrastructure Needs Repairing",
    #         "Number of rooms available in the health facilities? Number",
    #         "How many bed are available in this hospital?",
    #         "OPD service avaliable?", "Immunization Service Avaliable",
    #         "How many days, immunization service is provided?",
    #         "Laboraotry Service Avaliable", "ASRH (Adolescent Friendly Services) Service Avaliable",
    #         "Mental health Service Avaliable", "Substance abuse Service Avaliable", "Oral Health Service Avaliable"]
    #
    # X = train_data[cols]
    # y = train_data['Passed Threshold']
    # # Build a logreg and compute the feature importances
    # model = LogisticRegression()
    # # create the RFE model and select 8 attributes
    # rfe = RFE(model, 13)
    # rfe = rfe.fit(X, y)
    # # summarize the selection of the attributes
    # print('Selected features: %s' % list(X.columns[rfe.support_]))
    #
    # # -------------------------
    # rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
    # rfecv.fit(X, y)
    #
    # print("Optimal number of features: %d" % rfecv.n_features_)
    # print('Selected features: %s' % list(X.columns[rfecv.support_]))
    #
    # Selected_features = ["Does this health facility have its own building?", "Infrastructure Needs Repairing",
    #                      "Number of rooms available in the health facilities? Number",
    #                      "How many bed are available in this hospital?", "OPD service avaliable?",
    #                      "Immunization Service Avaliable", "How many days, immunization service is provided?",
    #                      "Laboraotry Service Avaliable", "ASRH (Adolescent Friendly Services) Service Avaliable",
    #                      "Mental health Service Avaliable", "Substance abuse Service Avaliable",
    #                      "Oral Health Service Avaliable"]
    #
    # X = train_data[Selected_features]
    # C = np.arange(1e-05, 5.5, 0.1)
    # scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}
    # log_reg = LogisticRegression()
    #
    # # Simple pre-processing estimators
    # ###############################################################################
    # std_scale = StandardScaler(with_mean=False, with_std=False)
    # # std_scale = StandardScaler()
    #
    # # Defining the CV method: Using the Repeated Stratified K Fold
    # ###############################################################################
    #
    # n_folds = 5
    # n_repeats = 5
    #
    # rskfold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=2)
    #
    # # Creating simple pipeline and defining the gridsearch
    # ###############################################################################
    #
    # log_clf_pipe = Pipeline(steps=[('scale', std_scale), ('clf', log_reg)])
    #
    # log_clf = GridSearchCV(estimator=log_clf_pipe, cv=rskfold,
    #                        scoring=scoring, return_train_score=True,
    #                        param_grid=dict(clf__C=C), refit='Accuracy')
    #
    # log_clf.fit(X, y)
    # results = log_clf.cv_results_
    #
    # # print('=' * 20)
    # print("best params: " + str(log_clf.best_estimator_))
    # print("best params: " + str(log_clf.best_params_))
    # print('best score:', (log_clf.best_score_) * 100)
    # # print('=' * 20)
    # test_data['Passed Threshold'] = log_clf.predict(test_data[Selected_features])
    # test_data['Ward No'] = test_df['Ward No']
    # test_data['Address'] = test_df['Address']
    # test_data['Type of Health facility'] = test_df['Type of Health facility']
    # test_data['long'] = test_df['long']
    # test_data['lat'] = test_df['lat']
    # submission = test_data[['Ward No', 'Address', 'Type of Health facility', 'Passed Threshold', 'long', 'lat']]
    # submission.to_csv("submission.csv", index=False)
    # submission.tail()
    dict = {}
    result = pd.read_csv("submission.csv")
    dict = {
        'ward': result['Ward No'],
        'address': result['Address'],
        'type': result['Type of Health facility'],
        'pass': result['Passed Threshold'],
        'long': result['long'],
        'lat': result['lat']
    }

    print(dict)
    return render(request, 'junapp/logistic.html')


def result(request):
    result = pd.read_csv("submission.csv")
    # print(result)
    a = result['Ward No']
    print(a)
    lines = result.split("\n")
    print(lines)
    # loop over the lines and save them in db. If error , store as string and then display
    # for line in lines:
    #     fields = line.split(",")
    #     data_dict = {}
    #     data_dict["name"] = fields[0]
    #     data_dict["start_date_time"] = fields[1]
    #     data_dict["end_date_time"] = fields[2]
    #     data_dict["notes"] = fields[3]
    return render(request, 'junapp/logistic.html')
