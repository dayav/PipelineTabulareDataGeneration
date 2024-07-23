from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.inspection import permutation_importance
from .base_evaluator import BaseEvaluator
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
import numpy as np
import rbo
from scipy import stats

from enum import Enum

class ClassifierType(Enum):
    LR = 'LR' 
    LDA = 'LDA'
    KNN = 'KNN' 
    CART = 'CART' 
    NB = 'GaussianNB' 
    SVM = 'SVM'
    XGBOOST = 'XGBoost'
    RANDOM_FOREST = 'Random Forest'


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if hasattr(X, 'toarray'):
            return X.toarray()
        else :
            return X
    
class MyLabelBinarizer(LabelEncoder):

    def fit(self, x, y=0):
        super().fit(x)
        return self
    
    def fit_transform(self, x, y=0):        
        return super().fit_transform(x)
    def transform(self, x, y=0):
        return super().transform(x)

class UtilityEvaluation(BaseEvaluator) :

    def __init__(self, real_train, synth , real_test, classifiers_ids) :
        super().__init__(real_train, synth)
        self._real_test = real_test
        self._classifiers_ids = classifiers_ids
        

    def get_model_name(self) :
        return [classifer[0] for classifer in  self._classifiers]

    def _get_model(self, classifier_id) :

        if classifier_id == ClassifierType.CART :
            return ('CART', DecisionTreeClassifier())
        elif classifier_id == ClassifierType.KNN :
            return ('KNN', KNeighborsClassifier())
        elif classifier_id == ClassifierType.LDA :
            return ('LDA', LinearDiscriminantAnalysis())
        elif classifier_id == ClassifierType.LR :
            return ('LR', LogisticRegression(solver='liblinear'))
        elif classifier_id == ClassifierType.NB :
            return ('NB', GaussianNB())
        elif classifier_id == ClassifierType.RANDOM_FOREST :
            return ('Random Forest', RandomForestClassifier(n_estimators=100))
        elif classifier_id == ClassifierType.SVM :
            return ('SVM', SVC())
        elif classifier_id == ClassifierType.XGBOOST :
            return ('XGBoost', XGBClassifier())
        elif classifier_id == ClassifierType.XGBOOST_ :
            return ('XGBoost_', XGBClassifier())
    
    def _get_data_target(self, data_df) :
        copy_data = data_df.copy(deep=True)
        y = copy_data.iloc[:,-1:].values
        X = copy_data.drop(copy_data.columns[-1], axis=1).values
        return X, y.ravel()
    
    def train_test_xgboost(self, isSyntheticData, with_preprocess=True, with_feature_importance=False, only_xgboost=False) :
        self._classifiers = [self._get_model(model_id) for model_id in self._classifiers_ids]
        if (isSyntheticData) :
            X_train, y_train = self._get_data_target(self._synth)
        else :
            X_train, y_train = self._get_data_target(self._real)

        X_real, y_real = self._get_data_target(self._real_test)

        #prepare target 
        le = LabelEncoder()
        le.fit(y_real)
        y_train_enc = le.transform(y_train)
        y_real_enc = le.transform(y_real)
            
            # define the data preparation for the columns
        categorical_idx = [self._real.columns.get_loc(cat_col) for cat_col in self._categorical_columns[0:-1]]
        numerical_idx = [self._real.columns.get_loc(cat_col) for cat_col in self._numerical_columns]
        t = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_idx), ('num', MinMaxScaler(), numerical_idx)]
        
        col_transform = ColumnTransformer(transformers=t)
        

        results = dict()
        p_is = dict()
        shap_values = dict()

        for name , model in  self._classifiers:
            if ((only_xgboost) and (name != 'XGBoost')) : continue
            if with_preprocess :
                if (name == 'LDA' or name == 'NB') :
                    pipeline = Pipeline(steps=[('prep',col_transform), ('to_dense', DenseTransformer()), ('m', model)])
                else :
                    pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])

            else :
                pipeline = Pipeline(steps=[('m', model)])

            pipeline.fit(X_train, y_train_enc)

            if ((with_feature_importance) and (name == 'XGBoost')):
                perm = permutation_importance(pipeline, X_train, y_train_enc, scoring='accuracy')
                shap_value = self.compute_shapvalues(model, X_train,categorical_idx, numerical_idx)
                
                shap_values[name] = np.abs(shap_value.values).mean(axis=0)
                p_is[name] = perm.importances_mean

            pred = pipeline.predict(X_real)
            results[name] = [accuracy_score(y_real_enc, pred), 
                             precision_score(y_real_enc, pred),
                             recall_score(y_real_enc, pred),
                             f1_score(y_real_enc, pred),
                             balanced_accuracy_score(y_real_enc, pred),
                             precision_score(y_real_enc, pred, average='weighted'),
                             recall_score(y_real_enc, pred, average='weighted'),
                             f1_score(y_real_enc, pred, average='weighted')]

        columns  = ['accuracy',
                    'precision',
                    'recall',
                    'f1', 
                    'balanced_accuracy', 
                    'precision_weighted',
                    'recall_weighted',
                    'f1_weighted']     
        accuracy_mean = sum([lst[3] for lst in results.values()]) / len(results)
        return pd.DataFrame.from_dict(results, orient='index', columns = columns), p_is,  shap_values, confusion_matrix(y_real_enc, pred), accuracy_mean
    
    def compute_shapvalues(self, model, X_train, categorical_idx, numerical_idx) :

        t_shap = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_idx), ('num', MinMaxScaler(), numerical_idx)]
        col_transform_shap = ColumnTransformer(transformers=t_shap)

        pipeline_shap = Pipeline(steps=[('prep',col_transform_shap)])
        X_train_shap = pipeline_shap.fit_transform(X_train)
        if not isinstance(X_train_shap, np.ndarray) :
            X_train_shap = X_train_shap.toarray()
        explainer = shap.Explainer(model)
        shap_value = explainer(X_train_shap)
        feature_names = self._real.columns
        n_categories = []
        for feat in feature_names[:-1]:
            if (feat in self._categorical_columns) :
                n = self._real[feat].nunique()
                n_categories.append(n)
            else : 
                n_categories.append(1)

        new_shap_values = []
        for values in shap_value.values:
            
            #split shap values into a list for each feature
            values_split = np.split(values , np.cumsum(n_categories))
            
            #sum values within each list
            values_sum = [sum(l) for l in values_split]
            
            new_shap_values.append(values_sum)

        
        #replace shap values
        shap_value.values = np.array(new_shap_values)

        #replace data with categorical feature values 
        new_data = X_train
        shap_value.data = np.array(new_data)

        #update feature names
        shap_value.feature_names = list(feature_names[:-1])

        return shap_value
   
    
    def rbo_compare_feature_importance(self, p_i_real, shap_value_real, p_i_synth, shap_value_synth) :

        importance_pi_real_df = pd.DataFrame([ self._real.columns[:-1], p_i_real]).T
        importance_pi_real_df.columns = ['column_name', 'shap_importance']
        importance_pi_real_df = importance_pi_real_df.sort_values('shap_importance', ascending=False)

        importance_pi_synth_df = pd.DataFrame([ self._real.columns[:-1], p_i_synth]).T
        importance_pi_synth_df.columns = ['column_name', 'shap_importance']
        importance_pi_synth_df = importance_pi_synth_df.sort_values('shap_importance', ascending=False)

        #--------------------------
        importance_shap_real_df = pd.DataFrame([ self._real.columns[:-1], shap_value_real]).T
        importance_shap_real_df.columns = ['column_name', 'shap_importance']
        importance_shap_real_df = importance_shap_real_df.sort_values('shap_importance', ascending=False)

        importance_shap_synth_df = pd.DataFrame([ self._real.columns[:-1], shap_value_synth]).T
        importance_shap_synth_df.columns = ['column_name', 'shap_importance']
        importance_shap_synth_df = importance_shap_synth_df.sort_values('shap_importance', ascending=False)

        return rbo.RankingSimilarity(importance_pi_real_df.index.to_list(), importance_pi_synth_df.index.to_list()).rbo(), rbo.RankingSimilarity(importance_shap_real_df.index.to_list(), importance_shap_synth_df.index.to_list()).rbo()

    def spearman_compare_feature_importance(self, p_i_real, shap_value_real, p_i_synth, shap_value_synth) :

        res_pi = stats.spearmanr(p_i_real, p_i_synth)
        res_shap = stats.spearmanr(shap_value_real, shap_value_synth)

        return res_pi.statistic, res_shap.statistic
    
    def kendall_compare_feature_importance(self, p_i_real, shap_value_real, p_i_synth, shap_value_synth) :

        res_pi = stats.kendalltau(p_i_real, p_i_synth)
        res_shap = stats.kendalltau(shap_value_real, shap_value_synth)

        return res_pi.statistic, res_shap.statistic

    
    def _train_test(self, isSyntheticData, with_feature_importance=False) :

        if (isSyntheticData) :
            X_train, y_train = self._get_data_target(self._synth)
        else :
            X_train, y_train = self._get_data_target(self._real)

        X_real, y_real = self._get_data_target(self._real_test)

        #prepare target 
        le = LabelEncoder()
        le.fit(y_real)
        y_train_enc = le.transform(y_train)
        y_real_enc = le.transform(y_real)

        # define the data preparation for the columns
        categorical_idx = [self._real.columns.get_loc(cat_col) for cat_col in self._categorical_columns[0:-1]]
        numerical_idx = [self._real.columns.get_loc(cat_col) for cat_col in self._numerical_columns]
        t = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_idx), ('num', MinMaxScaler(), numerical_idx)]
        # t = [('cat', OneHotEncoder(), categorical_idx), ('num', MinMaxScaler(), numerical_idx)]
        col_transform = ColumnTransformer(transformers=t)
        models = []
        models.append(('LR', LogisticRegression(solver='liblinear')))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        models.append(('XGBoost', XGBClassifier()))
        

        results = dict()
        f_i = dict()

        for name , model in models:
            if (name == 'LDA' or name == 'NB') :
                pipeline = Pipeline(steps=[('prep',col_transform), ('to_dense', DenseTransformer()), ('m', model)])
            else :
                pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])

            pipeline.fit(X_train, y_train_enc)

            if with_feature_importance :
                f_i[name] = permutation_importance(pipeline, X_train, y_train_enc, scoring='accuracy')

            pred = pipeline.predict(X_real)
            results[name] = [accuracy_score(y_real_enc, pred), 
                             precision_score(y_real_enc, pred, average='weighted'),
                             recall_score(y_real_enc, pred, average='weighted'),
                             f1_score(y_real_enc, pred, average='weighted')]
            

       
        return pd.DataFrame.from_dict(results, orient='index', columns = ['accuracy','precision','recall','f1']), pd.DataFrame.from_dict(f_i, orient='index', columns = self._real.columns)
    

    def _train_test_non_preprocess(self, isSyntheticData) :

        if (isSyntheticData) :
            X_train, y_train = self._get_data_target(self._synth)
        else :
            X_train, y_train = self._get_data_target(self._real)

        X_real, y_real = self._get_data_target(self._real_test)

        #prepare target 
        le = LabelEncoder()
        le.fit(y_real)
        y_train_enc = le.transform(y_train)
        y_real_enc = le.transform(y_real)

        # define the data preparation for the columns
        categorical_idx = [self._real.columns.get_loc(cat_col) for cat_col in self._categorical_columns[0:-1]]
        numerical_idx = [self._real.columns.get_loc(cat_col) for cat_col in self._numerical_columns]
        t = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_idx), ('num', MinMaxScaler(), numerical_idx)]
        # t = [('cat', OneHotEncoder(), categorical_idx), ('num', MinMaxScaler(), numerical_idx)]
        col_transform = ColumnTransformer(transformers=t)
        models = []
        models.append(('LR', LogisticRegression(solver='liblinear')))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
    
        test = col_transform.fit_transform(X_train)
        

        results = dict()
        accuracy = []
        precision = []
        recall = []
        f1 = []
        for name , model in models:
            # if (name == 'LDA' or name == 'NB') :
            #     pipeline = Pipeline(steps=[('prep',col_transform), ('to_dense', DenseTransformer()), ('m', model)])
            # else :
            #     pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])

            pipeline = Pipeline(steps=[('m', model)])
            pipeline.fit(X_train, y_train_enc)
            pred = pipeline.predict(X_real)
            results[name] = [accuracy_score(y_real_enc, pred), 
                             precision_score(y_real_enc, pred, average='weighted'),
                             recall_score(y_real_enc, pred, average='weighted'),
                             f1_score(y_real_enc, pred, average='weighted')]
            accuracy.append(accuracy_score(y_real_enc, pred))
            precision.append(precision_score(y_real_enc, pred, average='weighted'))
            recall.append(recall_score(y_real_enc, pred, average='weighted'))
            f1.append(f1_score(y_real_enc, pred, average='weighted'))
       
        return pd.DataFrame.from_dict(results, orient='index', columns = ['accuracy','precision','recall','f1'])



    def train_real_test_real(self, with_feature_importance=False) :
        return self.train_test_xgboost(False, True, with_feature_importance)
    
    def train_synthetic_test_real(self, with_feature_importance=False) :
        return self.train_test_xgboost(True, True, with_feature_importance)
    
    def train_real_test_real_with_xgboost(self, with_feature_importance=False) :

        return self.train_test_xgboost(False, True, False, True)
    
    def train_synthetic_test_real_xgboost(self, with_feature_importance=False) :
        return self.train_test_xgboost(True, True, False, True)

    def feature_importance_analysis(self, isSyntheticData) :
        return
        
