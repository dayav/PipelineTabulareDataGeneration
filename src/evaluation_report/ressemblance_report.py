import pickle
from typing import Dict, Optional

from data_synthesizer.pipeline.pipeline_results import PipelineResults
import ipywidgets as widgets
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, HTML

from data_evaluator.global_evaluator import GlobalEvaluator
from data_evaluator.multivariate_evaluator import MultivariateEvaluator
from data_evaluator.univariate_evaluator import UnivariateEvaluator
from data_evaluator.plot_evaluation import plot_correlation_diff, plot_correlation_ratio_diff


class ResemblanceReport :

    def __init__(self, real_data : pd.DataFrame, categorical_columns : list, numerical_columns : list, pipeline_results : Dict[str, PipelineResults], ) :
        self._pipeline_results = pipeline_results
        self._real_data = real_data
        self._categorical_columns = categorical_columns
        self._numerical_columns = numerical_columns
    
    def get_categorical_univariate_report(self) :
        accordions = []
        box_plot_data = {}
        kl_means = {}
        mode_collapse_values = {}
            
        for j_acc, j in zip(self._pipeline_results, range(len(self._pipeline_results))) :
            categorical_univariate_result = self._pipeline_results[j_acc]['resemblance_evaluation_results']['categorical_univariate']
            synthetic_data = self._pipeline_results[j_acc]['generation_results']['synthetic_data']


            chi_test = categorical_univariate_result['chi_test'].style.applymap(UnivariateEvaluator.style_negative, props='color:red;')
            jensen_shanon = categorical_univariate_result['jensen_shanon'].style.applymap(UnivariateEvaluator.style_binary, props='color:red;', subset= pd.IndexSlice[:, ['JS_divergence']])
            jensen_shanon_mean = categorical_univariate_result['jensen_shanon_data_mean']

            tabular_output_1 = widgets.HTML(chi_test.to_html())
            tabular_output_2 = widgets.HTML(jensen_shanon.to_html())
            tabular_output_3 = widgets.HTML(jensen_shanon_mean.to_frame().to_html())
            # tabular_output_4 = widgets.HTML(pd.DataFrame.from_dict(mode_collapse_values[j_acc], orient='index').to_html())
            box_plot_data[j_acc] = jensen_shanon.data['JS_divergence']
            kl_means[j_acc] = jensen_shanon_mean
            dropdown = widgets.Dropdown(options=self._categorical_columns, description='Columns:')
            output = widgets.interactive_output(self.display_dataframe_and_plot,
                                                    {'real':widgets.fixed(self._real_data),'synthetic': widgets.fixed(synthetic_data), 'column': dropdown})

            accordion = widgets.Accordion(children=[widgets.VBox([widgets.HBox([tabular_output_1, tabular_output_2, tabular_output_3]), dropdown, output])])
            accordion.set_title(0, j_acc)
            accordions.append(accordion)

        tabular_output_1 = widgets.Button(description="Show Boxplot")
        tabular_output_1.on_click(lambda b : self.plot_boxplot( box_plot_data, "JS divergence Boxplots comparison "))
        tabular_output_2 = widgets.HTML(pd.DataFrame(kl_means).transpose().to_html())
        tabular_output_3 = widgets.HTML(pd.DataFrame(mode_collapse_values).to_html())
        
        accordion = widgets.Accordion(children=[widgets.VBox([widgets.HBox([tabular_output_1, tabular_output_2, tabular_output_3])])])
        accordion.set_title(0, 'Summary')
        accordions.append(accordion)
        return widgets.VBox(accordions)
    
    # def build_numerical_univariate_report(self) :

    #     for j_acc, j in zip(self._evaluation_dict, range(len(self._evaluation_dict))) :
    #         self._uni_evaluators[j_acc].evaluate_numerical_stat_evaluation()
    #         self.num_uni[j_acc] = self._uni_evaluators[j_acc].num_uni


    # def get_numerical_univariate_report(self) :
    #     accordions = []
    #     box_plot_data = dict()
    #     cohens_means = dict()

    #     for j_acc, j in zip(self._evaluation_dict, range(len(self._evaluation_dict))) :
    #         eval = self._uni_evaluators[j_acc].get_numerical_stat_evaluation()
    #         tabular_output_1 = widgets.HTML(eval[0].to_html())
    #         tabular_output_2 = widgets.HTML(eval[1].to_html())
    #         tabular_output_3 = widgets.HTML(eval[2].to_frame().to_html())
    #         box_plot_data[j_acc] = eval[1].data['cohen_s_d']
    #         cohens_means[j_acc] = eval[2]
    #         dropdown = widgets.Dropdown(options=self._uni_evaluators[j_acc]._numerical_columns, description='Columns:')
    #         output = widgets.interactive_output(self.display_dataframe_and_plot,
    #                                                 {'real':widgets.fixed(self._evaluation_dict[j_acc][0]),'synth': widgets.fixed(self._evaluation_dict[j_acc][1]), 'column': dropdown})

    #         accordion = widgets.Accordion(children=[widgets.VBox([widgets.HBox([tabular_output_1, tabular_output_2, tabular_output_3]), dropdown, output])])
    #         accordion.set_title(0, j_acc)
    #         accordions.append(accordion)

    #     tabular_output_1 = widgets.Button(description="Show Boxplot")
    #     tabular_output_1.on_click(lambda b : self.plot_boxplot( box_plot_data, "Cohen's d Boxplots comparison "))
    #     tabular_output_2 = widgets.HTML(pd.DataFrame(cohens_means).transpose().to_html())
        
    #     accordion = widgets.Accordion(children=[widgets.VBox([widgets.HBox([tabular_output_1, tabular_output_2])])])
    #     accordion.set_title(0, 'Summary')
    #     accordions.append(accordion)
    #     return widgets.VBox(accordions)
    
    # def build_categorical_multivariate_report(self) :
    #     for j_acc, j in zip(self._evaluation_dict, range(len(self._evaluation_dict))) :
    #         self._multi_evaluator[j_acc].evaluate_cramerV_correlation_matrix_diff()
    #         self.cat_multi[j_acc] = self._multi_evaluator[j_acc].cat_multi

    
    # #TODO The same as univariate report
    # def get_categorical_multivariate_report(self) :
    #     sub_sub_tab=[widgets.Output() for j in range(len(self._evaluation_dict))]
    #     acc = widgets.Accordion(sub_sub_tab)
        
    #     for j_acc, j in zip(self._evaluation_dict, range(len(self._evaluation_dict))) :
    #         acc.set_title(j , j_acc) 

    #         with sub_sub_tab[j]:
    #             corr_real, corr_synth, norm = self._multi_evaluator[j_acc].get_cramerV_correlation_matrix_diff()
    #             plot_correlation_diff(corr_real, corr_synth, norm)
        
    #     return acc
    
    # def build_numerical_multivariate_report(self) :        
    #     for j_acc, j in zip(self._evaluation_dict, range( len(self._evaluation_dict))) :
    #         self._multi_evaluator[j_acc].evaluate_pearson_correlation_matrix_diff()
    #         self.num_multi[j_acc] =  self._multi_evaluator[j_acc].num_multi


    # def get_numerical_multivariate_report(self) :
    #     sub_sub_tab=[widgets.Output() for j in range(len(self._evaluation_dict))]
    #     acc = widgets.Accordion(sub_sub_tab)
        
    #     for j_acc, j in zip(self._evaluation_dict, range( len(self._evaluation_dict))) :
    #         acc.set_title(j , j_acc) 

    #         with sub_sub_tab[j]:
    #             corr_real, corr_synth, norm = self._multi_evaluator[j_acc].get_pearson_correlation_matrix_diff()
    #             plot_correlation_diff(corr_real, corr_synth, norm)

    #     return acc

    # def build_numcat_multivariate_report(self) :        
    #     for j_acc, j in zip(self._evaluation_dict, range( len(self._evaluation_dict))) :
    #         self._multi_evaluator[j_acc].evaluate_correlation_ratio_matrix_diff()
    #         self.num_cat_multi[j_acc] = self._multi_evaluator[j_acc].num_cat_multi
    
    # def get_numcat_multivariate_report(self) :
    #     sub_sub_tab=[widgets.Output() for j in range(len(self._evaluation_dict))]
    #     acc = widgets.Accordion(sub_sub_tab)
        
    #     for j_acc, j in zip(self._evaluation_dict, range( len(self._evaluation_dict))) :
    #         acc.set_title(j , j_acc) 

    #         with sub_sub_tab[j]:
    #             corr_real, corr_synth, norm = self._multi_evaluator[j_acc].get_correlation_ratio_matrix_diff()
    #             plot_correlation_ratio_diff(corr_real, corr_synth, norm)

    #     return acc

    # def build_identification_evaluation(self) :
    #     sub_sub_tab=[widgets.Output() for j in range(len(self._evaluation_dict))]
    #     acc = widgets.Accordion(sub_sub_tab)
        
    #     for j_acc, j in zip(self._evaluation_dict, range(len(self._evaluation_dict))) :
    #         acc.set_title(j , j_acc) 

    #         with sub_sub_tab[j]:
    #             pMSE, _ = self._global_evaluator[j_acc].propensity_score()
    #             print('pMSE : ', pMSE[1])
    #             accs, names = self._global_evaluator[j_acc].model_identification_accuracy()
    #             self._box_plot_results(accs, 'Accuracy identification Algo', names)
        
    #     return acc
    
    def display_dataframe_and_plot(self, real, synthetic, column):
        
        # Display a horizontal line as a ribbon
        display(HTML('<hr>'))
        
        # Plot the distribution
        self.plot_distribution(real, synthetic, column)

    def plot_distribution(self, real, df, column):
        if df[column].dtype == 'float64':
            # Plot histogram
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            real[column].plot(kind='hist', density=True, alpha=0.7)
            plt.title('Histogram real')
            
            plt.subplot(1, 2, 2)
            df[column].plot(kind='hist', density=True, alpha=0.7)
            plt.title('Histogram synthetic')
            plt.show()
        else:
            # For categorical data, just show a bar plot
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            real[column].value_counts().plot(kind='bar')
            plt.title('Histogram real')

            plt.subplot(1, 2, 2)
            df[column].value_counts().plot(kind='bar')
            plt.title('Histogram synthetic')
            plt.show()

    # def plot_distribution(self, real, df, column):
    #     if df[column].dtype == 'float64':
    #         # Plot histogram
    #         plt.figure(figsize=(12, 5))
    #         plt.subplot(1, 2, 1)
    #         real[column].plot(kind='hist', density=True, alpha=0.7)
    #         plt.title('Histogram real')
            
    #         plt.subplot(1, 2, 2)
    #         df[column].plot(kind='hist', density=True, alpha=0.7)
    #         plt.title('Histogram synthetic')
    #         plt.show()
    #     else:
    #         # For categorical data, just show a bar plot
    #         plt.figure(figsize=(12, 5))
    #         plt.subplot(1, 2, 1)
    #         real[column].value_counts().plot(kind='bar')
    #         plt.title('Histogram real')

    #         plt.subplot(1, 2, 2)
    #         df[column].value_counts().plot(kind='bar')
    #         plt.title('Histogram synthetic')
    #         plt.show()

    # def save_evaluation(self, file_name) :

    #     saved = {'evaluation_dict' : self._evaluation_dict,
    #             'uni_evaluators' : self._uni_evaluators , 
    #             'multi_evaluator' : self._multi_evaluator,
    #             'global_evaluator' : self._global_evaluator,
    #             'cat_uni' : self.cat_uni,
    #             'num_uni' : self.num_uni,
    #             'cat_multi' : self.cat_multi,
    #             'num_multi' : self.num_multi,
    #             'num_cat_multi' : self.num_cat_multi}
        
    #     with open(file_name +'.pkl', 'wb') as file:
    #         pickle.dump(saved, file)

    # @staticmethod
    # def load_evaluation(file_name) :
    #     with open(file_name, 'rb') as file:
    #         loaded_object = pickle.load(file)

    #     if 'num_cat_multi' in loaded_object :  
    #         report = ResemblanceReport(evaluation_dict=None, loaded=loaded_object)
    #     else :
    #         report = None

    #     return report



