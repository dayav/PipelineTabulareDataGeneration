import ipywidgets as widgets
from IPython.display import display
from .ressemblance_report import ResemblanceReport
from .privacy_report import PrivacyReport
from .privacy_anonymeter_report import PrivacyAnonymeterReport


class EvaluationReport() :

    def __init__(self , ressemblance_report, utility, privacy, privacy_anonymeter) :
        self.ressemblance_report = ressemblance_report
        self.utility = utility
        self.privacy = privacy
        self.privacy_anonymeter = privacy_anonymeter
        self.test = {'a': 1,'b': 2 }

    def display_evaluation(self) :
        tab = widgets.Tab()
        children = []

        if self.ressemblance_report != None : 
            children.append(self.ressemblance_report.get_categorical_univariate_report())
            children.append(self.ressemblance_report.get_numerical_univariate_report())
            children.append(self.ressemblance_report.get_categorical_multivariate_report())
            children.append(self.ressemblance_report.get_numerical_multivariate_report())
            children.append(self.ressemblance_report.get_numcat_multivariate_report())

        if self.utility != None : children.append(self.utility.get_report())        
        if self.privacy != None : children.append(self.privacy.get_report())
        if self.privacy_anonymeter != None : children.append(self.privacy_anonymeter.get_report())

        tab.children = children
        display(tab)
    
    def save(self, file_name) :
        if self.ressemblance_report != None : self.ressemblance_report.save_evaluation(file_name)
        if self.privacy != None : self.privacy.save_evaluation(file_name)
        if self.privacy_anonymeter != None : self.privacy_anonymeter.save_evaluation(file_name)


    @staticmethod
    def load(file_name) :
        ressemblance_report = ResemblanceReport.load_evaluation(file_name)
        privacy_report = PrivacyReport.load_evaluation(file_name)
        privacy_anonymeter = PrivacyAnonymeterReport.load_evaluation(file_name)
        report = EvaluationReport(ressemblance_report, None, privacy_report, privacy_anonymeter)
        return report