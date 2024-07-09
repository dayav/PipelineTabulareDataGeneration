import pickle
import pandas as pd
import ipywidgets as widgets
from .privacy_report import PrivacyReport

from data_evaluator.privacy_evaluation import PrivacyEvaluatorAnonymeter

class PrivacyAnonymeterReport :
    
    def __init__(self, evaluation_dict, data_test = None, loaded = None) :
        
        if (evaluation_dict != None) :         
            self._evaluation_dict = evaluation_dict
            self._privacy_evaluator_anonymeter = {}
            for name in evaluation_dict :
                self._privacy_evaluator_anonymeter[name] = PrivacyEvaluatorAnonymeter(evaluation_dict[name][0],
                                                                                    evaluation_dict[name][1], 
                                                                                    data_test)
        
        elif (loaded != None) :
            self._evaluation_dict = loaded['evaluation_dict']
            self._privacy_evaluator_anonymeter = loaded['privacy_evaluator_anonymeter']
            self.result_singling_uni = loaded['result_singling_uni']
            self.result_singling_multi = loaded['result_singling_multi']
            self.result_linkability_multi = loaded['result_linkability_multi']        
            

    def build_report(self) :

        self.result_singling_uni = {'attacks_numbers' : [], 
                                'attacks_succeeded': [],
                                'privacy_risk_original': [],
                                'privacy_risk_control': [],
                                'privacy_risk_naive': [],
                                'specific_privacy': [] }
        
        self.result_singling_multi = {'attacks_numbers' : [], 
                        'attacks_succeeded': [],
                        'privacy_risk_original': [],
                        'privacy_risk_control': [],
                        'privacy_risk_naive': [],
                        'specific_privacy': [] }
        
        self.result_linkability_multi = {'attacks_numbers' : [], 
                        'privacy_risk_original': [],
                        'privacy_risk_control': [],
                        'privacy_risk_naive': [],
                        'specific_privacy': [] }

        for j_acc, j in zip(self._evaluation_dict, range(len(self._evaluation_dict))) :          

            self._privacy_evaluator_anonymeter[j_acc].configure_singling_uni_attacks()
            result = None
            while result is None : 
                result = self._privacy_evaluator_anonymeter[j_acc].evaluate_singling_uni_attacks()
                print('again', j_acc)
            self.result_singling_uni['attacks_numbers'].append(result.attacks_numbers)
            self.result_singling_uni['attacks_succeeded'].append(result.attacks_succeeded)
            self.result_singling_uni['privacy_risk_original'].append(result.privacy_risk_original)
            self.result_singling_uni['privacy_risk_control'].append(result.privacy_risk_control)
            self.result_singling_uni['privacy_risk_naive'].append(result.privacy_risk_naive)
            self.result_singling_uni['specific_privacy'].append(result.specific_privacy)

            self._privacy_evaluator_anonymeter[j_acc].configure_singling_multi_attacks()
            result = self._privacy_evaluator_anonymeter[j_acc].evaluate_singling_multi_attacks()

            self.result_singling_multi['attacks_numbers'].append(result.attacks_numbers)
            self.result_singling_multi['attacks_succeeded'].append(result.attacks_succeeded)
            self.result_singling_multi['privacy_risk_original'].append(result.privacy_risk_original)
            self.result_singling_multi['privacy_risk_control'].append(result.privacy_risk_control)
            self.result_singling_multi['privacy_risk_naive'].append(result.privacy_risk_naive)
            self.result_singling_multi['specific_privacy'].append(result.specific_privacy)

            self._privacy_evaluator_anonymeter[j_acc].configure_linkability_attacks()
            result = self._privacy_evaluator_anonymeter[j_acc].evaluate_linkability_attacks()

            self.result_linkability_multi['attacks_numbers'].append(result.attacks_numbers)
            self.result_linkability_multi['privacy_risk_original'].append(result.privacy_risk_original)
            self.result_linkability_multi['privacy_risk_control'].append(result.privacy_risk_control)
            self.result_linkability_multi['privacy_risk_naive'].append(result.privacy_risk_naive)
            self.result_linkability_multi['specific_privacy'].append(result.specific_privacy)



    def get_report(self) :
        accordions = []

        result_singling_uni_df = pd.DataFrame.from_dict({'Attacks numbers NA' : self.result_singling_uni['attacks_numbers'],
                                                         'Attacks succeeded' : self.result_singling_uni['attacks_succeeded'],
                                                         'Privacy Risk original' : self.result_singling_uni['privacy_risk_original'],
                                                         'Privacy Risk control' : self.result_singling_uni['privacy_risk_control'],
                                                         'Privacy Risk naive' : self.result_singling_uni['privacy_risk_naive'],
                                                         'Specific Privacy ' : self.result_singling_uni['specific_privacy']}, orient='index', columns=self._evaluation_dict.keys())
        
        result_singling_multi_df = pd.DataFrame.from_dict({'Attacks numbers NA' : self.result_singling_multi['attacks_numbers'],
                                                    'Attacks succeeded' : self.result_singling_multi['attacks_succeeded'],
                                                    'Privacy Risk original' : self.result_singling_multi['privacy_risk_original'],
                                                    'Privacy Risk control' : self.result_singling_multi['privacy_risk_control'],
                                                    'Privacy Risk naive' : self.result_singling_multi['privacy_risk_naive'],
                                                    'Specific Privacy ' : self.result_singling_multi['specific_privacy']}, orient='index', columns=self._evaluation_dict.keys())

        result_linkability_df = pd.DataFrame.from_dict({'Attacks numbers NA' : self.result_linkability_multi['attacks_numbers'],
                                                    'Privacy Risk original' : self.result_linkability_multi['privacy_risk_original'],
                                                    'Privacy Risk control' : self.result_linkability_multi['privacy_risk_control'],
                                                    'Privacy Risk naive' : self.result_linkability_multi['privacy_risk_naive'],
                                                    'Specific Privacy ' : self.result_linkability_multi['specific_privacy']}, orient='index', columns=self._evaluation_dict.keys())

        tabular_output_1 = widgets.HTML(result_singling_uni_df.to_html())
        accordion_1 = widgets.Accordion(children=[widgets.VBox([tabular_output_1])])
        accordion_1.set_title(0, 'singling univariate')
        accordions.append(accordion_1)

 
        tabular_output_2 = widgets.HTML(result_singling_multi_df.to_html())
        accordion_2 = widgets.Accordion(children=[widgets.VBox([tabular_output_2])])
        accordion_2.set_title(0, 'singling multivariate')
        accordions.append(accordion_2)     

        tabular_output_3 = widgets.HTML(result_linkability_df.to_html())
        accordion_3 = widgets.Accordion(children=[widgets.VBox([tabular_output_3])])
        accordion_3.set_title(0, 'linkability')
        accordions.append(accordion_3)      

        return widgets.VBox(accordions)
    
    def save_evaluation(self, file_name) :

        saved = {'evaluation_dict' : self._evaluation_dict,
                'privacy_evaluator_anonymeter' : self._privacy_evaluator_anonymeter , 
                'result_singling_uni' : self.result_singling_uni,
                'result_singling_multi' : self.result_singling_multi,
                'result_linkability_multi' : self.result_linkability_multi}
        
        with open(file_name +'.pkl', 'wb') as file:
            pickle.dump(saved, file)

    @staticmethod
    def load_evaluation(file_name) :
        with open(file_name, 'rb') as file:
            loaded_object = pickle.load(file)
        
        if 'result_linkability_multi' in loaded_object : 
            report = PrivacyAnonymeterReport(evaluation_dict=None, loaded=loaded_object)
        else : 
            report = None

        return report

