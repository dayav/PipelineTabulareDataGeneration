
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import ipywidgets as widgets

from .ressemblance_report import ResemblanceReport
from .utility_report import UtilityReport
from .privacy_report import PrivacyReport
from .privacy_anonymeter_report import PrivacyAnonymeterReport
from .evaluation_report import EvaluationReport




from enum import Enum

class EvaluationType(Enum):
    CATEGORIAL_UNIVARIATE_TAB = 0 
    NUMERICAL_UNIVARIATE_TAB = 1 
    CATEGORIAL_MULTIVARIATE_TAB = 2 
    NUMERICAL_MULTIVARIATE_TAB = 3 
    NUMERICAL_CATEGORICAL_MULTIVARIATE_TAB = 4  
    IDENTIFICATION_ACCURACY_TAB = 5 
    UTILITY_EVALUATION_TAB = 6
    PRIVACY_EVALUATION_TAB = 7
    PRIVACY_ANONYMETER_EVALUATION_TAB = 8


class EvaluationReportBuilder() :

    def __init__(self , evaluation_dict) :
        self._evaluation_dict = evaluation_dict
        self._resemblance_report = None
        self._utility_report = None
        self._privacy_report = None
        self._privacy_anonymeter_report = None

        self.cat_uni = None
        self.num_uni = None
        self.cat_multi = None
        self.num_multi = None
        self.num_cat_multi = None

        self.utility = None

        self.privacy = None

        self.privacy_anonymeter = None

    def with_resemblance(self) :
        self._resemblance_report = ResemblanceReport(self._evaluation_dict)
        return self

    def with_utility(self, data_test, classifier_utility) :
        self._utility_report = UtilityReport(self._evaluation_dict, data_test, classifier_utility)
        return self
      
    def with_privacy(self, data_test, qid_columns, non_quid_columns, dissimilarities) :
        self._privacy_report = PrivacyReport(self._evaluation_dict, data_test, qid_columns, non_quid_columns, dissimilarities = dissimilarities)
        return self
    
    def with_privacy_anonymeter(self, data_test) :
        self._privacy_anonymeter_report = PrivacyAnonymeterReport(self._evaluation_dict, data_test)
        return self
    
    def build(self) : 
        if self._resemblance_report != None :
            self._resemblance_report.build_categorical_univariate_report()
            self._resemblance_report.build_numerical_univariate_report()
            self._resemblance_report.build_categorical_multivariate_report()
            self._resemblance_report.build_numerical_multivariate_report()
            self._resemblance_report.build_numcat_multivariate_report()

        if self._utility_report != None :
            self._utility_report.build_report()

        if self._privacy_report != None :
            self._privacy_report.build_report() 

        if self._privacy_anonymeter_report != None :
            self._privacy_anonymeter_report.build_report()

        return EvaluationReport(self._resemblance_report,
                                self._utility_report,
                                self._privacy_report,
                                self._privacy_anonymeter_report)
    
    