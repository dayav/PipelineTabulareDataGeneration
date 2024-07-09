import ipywidgets as widgets
import pandas as pd
from IPython.display import display, HTML
import plotly.graph_objects as go

from data_evaluator.utility_evaluator import UtilityEvaluation, ClassifierType
from data_evaluator.plot_evaluation import histo_plot_utility_compare


class UtilityReport :
    
    def __init__(self, evaluation_dict, data_test, classifier_utility) :
        self._evaluation_dict = evaluation_dict
        self._utility_evaluator = {}
        for name in evaluation_dict :
            self._utility_evaluator[name] = UtilityEvaluation(evaluation_dict[name][0], 
                                                              evaluation_dict[name][1], 
                                                              data_test, 
                                                              classifier_utility)
            
    
    def build_report(self) : 
        # output_1 = widgets.Output()
        # output_2 = widgets.Output()
        
        # with output_1:
        self.plot = {}
        self.f_i = dict()
        self.shap_importances = dict()
        acc_means = dict()
        for utility_key in self._utility_evaluator :
            synth, importance, shap_importance, mat, acc_mean = self._utility_evaluator[utility_key].train_synthetic_test_real(True)
            self.plot[utility_key] = synth
            self.f_i[utility_key] = importance
            self.shap_importances[utility_key] = shap_importance
            acc_means[utility_key] = acc_mean
        real, importance, shap_importance, mat, acc_mean = self._utility_evaluator[utility_key].train_real_test_real(True)
        acc_means['real'] = acc_mean
        acc_means = { key: abs(value - acc_mean) for key, value in acc_means.items()}
        self.plot['real'] = real
        self.acc_mean = pd.DataFrame(acc_means, index=[0])


        self.f_i['real'] = importance
        self.shap_importances['real'] = shap_importance
        features_key = list(self._evaluation_dict)[0]
        self.f_i['variable'] = self._evaluation_dict[features_key][0].columns[:-1]
        self.shap_importances['variable'] = self._evaluation_dict[features_key][0].columns[:-1]
            
    #---------------------------------------------------------------------------------           
        first_key = list(self._utility_evaluator.keys())[0]
        model_names = self._utility_evaluator[first_key].get_model_name()
        sub_sub_sub_tab = [widgets.Output() for x in model_names]
        acc_sub_1 = widgets.Accordion(sub_sub_sub_tab)

        for id, model_name in zip(range(len(model_names)), model_names) :
            acc_sub_1.set_title(id, model_name)
            with sub_sub_sub_tab[id] :
                f_i_dict = dict()
                f_i_dict['real'] = self.f_i['real'][model_name]
                for utility_key in self._utility_evaluator :                            
                    f_i_dict[utility_key] = self.f_i[utility_key][model_name]
                display(pd.DataFrame(f_i_dict).sort_values(by=['real'], ascending=False))
    #---------------------------------------------------------------------------------
        first_key = list(self._utility_evaluator.keys())[0]
        model_names = self._utility_evaluator[first_key].get_model_name()
        sub_sub_sub_tab = [widgets.Output() for x in model_names]
        acc_sub_2 = widgets.Accordion(sub_sub_sub_tab)
        for id, model_name in zip(range(len(model_names)), model_names) :
            acc_sub_2.set_title(id, model_name)
            with sub_sub_sub_tab[id] :
                scatters = []
                for utility_key in self._utility_evaluator :
                    scatters.append(go.Scatter( mode='lines+markers', x=self.f_i['variable'], y=self.f_i[utility_key][model_name], name=utility_key))
                scatters.append(go.Scatter( mode='lines+markers', x=self.f_i['variable'], y=self.f_i['real'][model_name], name='real'))
                fig = go.FigureWidget(data=scatters)
                display(fig)
    #---------------------------------------------------------------------------------
        first_key = list(self._utility_evaluator.keys())[0]
        model_names = self._utility_evaluator[first_key].get_model_name()
        sub_sub_sub_tab = [widgets.Output() for x in model_names]
        acc_sub_3 = widgets.Accordion(sub_sub_sub_tab)
        for id, model_name in zip(range(len(model_names)), model_names) :
            acc_sub_3.set_title(id, model_name)
            with sub_sub_sub_tab[id] :
                shap_importances_dict = dict()
                shap_importances_dict['real'] = self.shap_importances['real'][model_name]
                for utility_key in self._utility_evaluator :                            
                    shap_importances_dict[utility_key] = self.shap_importances[utility_key][model_name]
                display(pd.DataFrame(shap_importances_dict).sort_values(by=['real'], ascending=False))
    #---------------------------------------------------------------------------------
        
        first_key = list(self._utility_evaluator.keys())[0]
        model_names = self._utility_evaluator[first_key].get_model_name()
        sub_sub_sub_tab = [widgets.Output() for x in model_names]
        acc_sub_4 = widgets.Accordion(sub_sub_sub_tab)
        for id, model_name in zip(range(len(model_names)), model_names) :
            acc_sub_4.set_title(id, model_name)
            with sub_sub_sub_tab[id] :
                scatters = []
                for utility_key in self._utility_evaluator :
                    scatters.append(go.Scatter( mode='lines+markers', x=self.shap_importances['variable'], y=self.shap_importances[utility_key][model_name], name=utility_key))
                scatters.append(go.Scatter( mode='lines+markers', x=self.shap_importances['variable'], y=self.shap_importances['real'][model_name], name='real'))
                fig = go.FigureWidget(data=scatters)
                display(fig)
    #---------------------------------------------------------------------------------
                

        correlation_values = dict() 
        for utility_key in self._utility_evaluator :
            rbo_pi, rbo_shap = self._utility_evaluator[utility_key].rbo_compare_feature_importance(self.f_i['real']['XGBoost'],
                                                                                                    self.shap_importances['real']['XGBoost'],
                                                                                                    self.f_i[utility_key]['XGBoost'],
                                                                                                    self.shap_importances[utility_key]['XGBoost'])
            correlation_values[utility_key] = [rbo_pi, rbo_shap]
            
        self.rbo_df = pd.DataFrame.from_dict(correlation_values, orient='index', columns = ['rbo permutation-importance', 'rbo shap-importance'])

    
    def get_report(self) : 
        output_1 = widgets.Output()
        output_2 = widgets.Output()
        
        with output_1:
            plot = {}
            f_i = dict()
            shap_importances = dict()
            acc_means = dict()
            for utility_key in self._utility_evaluator :
                synth, importance, shap_importance, mat, acc_mean = self._utility_evaluator[utility_key].train_synthetic_test_real(True)
                plot[utility_key] = synth
                f_i[utility_key] = importance
                shap_importances[utility_key] = shap_importance
                acc_means[utility_key] = acc_mean
            real, importance, shap_importance, mat, acc_mean = self._utility_evaluator[utility_key].train_real_test_real(True)
            acc_means['real'] = acc_mean
            acc_means = { key: abs(value - acc_mean) for key, value in acc_means.items()}
            plot['real'] = real
            display(pd.DataFrame(acc_means, index=[0]))
            histo_plot_utility_compare(plot)

            f_i['real'] = importance
            shap_importances['real'] = shap_importance
            features_key = list(self._evaluation_dict)[0]
            f_i['variable'] = self._evaluation_dict[features_key][0].columns[:-1]
            shap_importances['variable'] = self._evaluation_dict[features_key][0].columns[:-1]
            
    #---------------------------------------------------------------------------------           
        first_key = list(self._utility_evaluator.keys())[0]
        model_names = self._utility_evaluator[first_key].get_model_name()
        sub_sub_sub_tab = [widgets.Output() for x in model_names]
        acc_sub_1 = widgets.Accordion(sub_sub_sub_tab)

        for id, model_name in zip(range(len(model_names)), model_names) :
            acc_sub_1.set_title(id, model_name)
            with sub_sub_sub_tab[id] :
                f_i_dict = dict()
                f_i_dict['real'] = f_i['real'][model_name]
                for utility_key in self._utility_evaluator :                            
                    f_i_dict[utility_key] = f_i[utility_key][model_name]
                display(pd.DataFrame(f_i_dict).sort_values(by=['real'], ascending=False))
    #---------------------------------------------------------------------------------
        first_key = list(self._utility_evaluator.keys())[0]
        model_names = self._utility_evaluator[first_key].get_model_name()
        sub_sub_sub_tab = [widgets.Output() for x in model_names]
        acc_sub_2 = widgets.Accordion(sub_sub_sub_tab)
        for id, model_name in zip(range(len(model_names)), model_names) :
            acc_sub_2.set_title(id, model_name)
            with sub_sub_sub_tab[id] :
                scatters = []
                for utility_key in self._utility_evaluator :
                    scatters.append(go.Scatter( mode='lines+markers', x=f_i['variable'], y=f_i[utility_key][model_name], name=utility_key))
                scatters.append(go.Scatter( mode='lines+markers', x=f_i['variable'], y=f_i['real'][model_name], name='real'))
                fig = go.FigureWidget(data=scatters)
                display(fig)
    #---------------------------------------------------------------------------------
        first_key = list(self._utility_evaluator.keys())[0]
        model_names = self._utility_evaluator[first_key].get_model_name()
        sub_sub_sub_tab = [widgets.Output() for x in model_names]
        acc_sub_3 = widgets.Accordion(sub_sub_sub_tab)
        for id, model_name in zip(range(len(model_names)), model_names) :
            acc_sub_3.set_title(id, model_name)
            with sub_sub_sub_tab[id] :
                shap_importances_dict = dict()
                shap_importances_dict['real'] = shap_importances['real'][model_name]
                for utility_key in self._utility_evaluator :                            
                    shap_importances_dict[utility_key] = shap_importances[utility_key][model_name]
                display(pd.DataFrame(shap_importances_dict).sort_values(by=['real'], ascending=False))
    #---------------------------------------------------------------------------------
        
        first_key = list(self._utility_evaluator.keys())[0]
        model_names = self._utility_evaluator[first_key].get_model_name()
        sub_sub_sub_tab = [widgets.Output() for x in model_names]
        acc_sub_4 = widgets.Accordion(sub_sub_sub_tab)
        for id, model_name in zip(range(len(model_names)), model_names) :
            acc_sub_4.set_title(id, model_name)
            with sub_sub_sub_tab[id] :
                scatters = []
                for utility_key in self._utility_evaluator :
                    scatters.append(go.Scatter( mode='lines+markers', x=shap_importances['variable'], y=shap_importances[utility_key][model_name], name=utility_key))
                scatters.append(go.Scatter( mode='lines+markers', x=shap_importances['variable'], y=shap_importances['real'][model_name], name='real'))
                fig = go.FigureWidget(data=scatters)
                display(fig)
    #---------------------------------------------------------------------------------
                
        with output_2 :
            rbo_values = dict() 
            for utility_key in self._utility_evaluator :
                rbo_pi, rbo_shap = self._utility_evaluator[utility_key].rbo_compare_feature_importance(f_i['real']['XGBoost'],
                                                                                                        shap_importances['real']['XGBoost'],
                                                                                                        f_i[utility_key]['XGBoost'],
                                                                                                        shap_importances[utility_key]['XGBoost'])
                rbo_values[utility_key] = [rbo_pi, rbo_shap]
                
            display(pd.DataFrame.from_dict(rbo_values, orient='index', columns = ['rbo permutation-importance', 'rbo shap-importance']))

            
            correlation_values = dict() 
            for utility_key in self._utility_evaluator :
                rbo_pi, rbo_shap = self._utility_evaluator[utility_key].rbo_compare_feature_importance(self.f_i['real']['XGBoost'],
                                                                                                        self.shap_importances['real']['XGBoost'],
                                                                                                        self.f_i[utility_key]['XGBoost'],
                                                                                                        self.shap_importances[utility_key]['XGBoost'])
                
                spearman_pi, spearman_shap = self._utility_evaluator[utility_key].spearman_compare_feature_importance(self.f_i['real']['XGBoost'],
                                                                                        self.shap_importances['real']['XGBoost'],
                                                                                        self.f_i[utility_key]['XGBoost'],
                                                                                        self.shap_importances[utility_key]['XGBoost'])
                
                kendall_pi, kendall_shap = self._utility_evaluator[utility_key].kendall_compare_feature_importance(self.f_i['real']['XGBoost'],
                                                                                        self.shap_importances['real']['XGBoost'],
                                                                                        self.f_i[utility_key]['XGBoost'],
                                                                                        self.shap_importances[utility_key]['XGBoost'])
                correlation_values[utility_key] = [rbo_pi, rbo_shap, spearman_pi, spearman_shap, kendall_pi, kendall_shap]
                
            display(pd.DataFrame.from_dict(correlation_values, orient='index', columns = ['rbo permutation-importance', 'rbo shap-importance', 
                                                                                                'spearman permutation-importance', 'spearman shap-importance', 
                                                                                                'kendall permutation-importance', 'kendall shap-importance']))
            
        acc = widgets.Accordion([output_1, acc_sub_1, acc_sub_2, acc_sub_3, acc_sub_4, output_2])
        acc.set_title(0 , 'bar comparison') 
        acc.set_title(1 , 'permutation feature importance') 
        acc.set_title(2 , 'permutation feature importance plot') 
        acc.set_title(3 , 'shap feature importance') 
        acc.set_title(4 , 'shap feature importance plot') 
        acc.set_title(5 , 'rbo evaluation feature importance')        

        return acc
