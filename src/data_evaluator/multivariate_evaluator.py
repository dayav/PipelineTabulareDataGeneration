import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats.contingency import association
from .base_evaluator import BaseEvaluator
from dython.nominal import correlation_ratio, associations

class MultivariateEvaluator(BaseEvaluator) :

    def __init__(self, real, synth) :
        super().__init__(real, synth)
        self.num_multi = None
        self.cat_multi = None
        self.num_cat_multi = None

   
    def _get_pearson_correlation_matrix(self, dataframe) :
        #compute the pearson pairwise correlation matrix of numerical attributes of the dataset
        cors = np.absolute(dataframe.corr(method='pearson', numeric_only=True))

        #compute the norm of the pearson pairwise correlation matrix computed before
        cors_norm = np.round(np.linalg.norm(cors),4)

        #return the values
        return cors, cors_norm
    
    def evaluate_pearson_correlation_matrix_diff(self):
        
        corr_real , real_norm = self._get_pearson_correlation_matrix(self._real)
        corr_synth , synth_norm = self._get_pearson_correlation_matrix(self._synth)

        self.pearson_real = corr_real
        self.pearson_synth = corr_synth
        self.pearson_norm_diff = abs(real_norm - synth_norm)

        self.num_multi = {'pearson_real' : corr_real, 'pearson_synth' : corr_synth, 'pearson_norm_diff' : abs(real_norm - synth_norm)}
    
    def get_pearson_correlation_matrix_diff(self):
        return self.num_multi['pearson_real'], self.num_multi['pearson_synth'] , self.num_multi['pearson_norm_diff']
          
    
    def get_cramerV_correlation_matrix(self, df):

        cramer = pd.DataFrame(
        np.eye(len(self._categorical_columns)),
        columns=self._categorical_columns,
        index=self._categorical_columns)

        all_combinations = combinations(self._categorical_columns, r=2)

        for comb in all_combinations:
            i = comb[0]
            j = comb[1]

            input_tab = pd.crosstab(df[i], df[j])

            res_cramer = association(input_tab, method='cramer')
            cramer[i][j], cramer[j][i] = res_cramer, res_cramer

        cors_norm = np.round(np.linalg.norm(cramer),4)

        return cramer, cors_norm
    
    def get_correlation_ratio_matrix(self, df):
        
        correlation_matrix = associations(df, 
              nominal_columns=self._categorical_columns.to_list(),
              numerical_columns=self._numerical_columns.to_list(),
              hide_rows=self._categorical_columns.to_list(),
               hide_columns=self._numerical_columns.to_list(),
               plot=False
              )


        cors_norm = np.round(np.linalg.norm(correlation_matrix['corr']),4)

        return correlation_matrix, cors_norm
    
    def evaluate_cramerV_correlation_matrix_diff(self) :
        cramer_real, real_norm = self.get_cramerV_correlation_matrix(self._real)
        cramer_synth, synth_norm = self.get_cramerV_correlation_matrix(self._synth)

        self.cramer_real = cramer_real
        self.cramer_synth = cramer_synth
        self.diff_norm_cramer = abs(real_norm - synth_norm)

        self.cat_multi = {'cramer_real' : cramer_real, 'cramer_synth' : cramer_synth, 'diff_norm_cramer' : abs(real_norm - synth_norm) }
    
    def get_cramerV_correlation_matrix_diff(self):
        return self.cat_multi['cramer_real'], self.cat_multi['cramer_synth'] , self.cat_multi['diff_norm_cramer']
    
    def evaluate_correlation_ratio_matrix_diff(self) :
        corr_ratio_real, real_norm = self.get_correlation_ratio_matrix(self._real)
        corr_ratio_synth, synth_norm = self.get_correlation_ratio_matrix(self._synth)

        self.corr_ratio_real = corr_ratio_real
        self.corr_ratio_synth = corr_ratio_synth
        self.diff_norm_corr_ratio = abs(real_norm - synth_norm)

        self.num_cat_multi = {'corr_ratio_real' : corr_ratio_real, 'corr_ratio_synth': corr_ratio_synth, 'diff_norm_corr_ratio' : abs(real_norm - synth_norm)}
    
    def get_correlation_ratio_matrix_diff(self) :       
        return self.num_cat_multi['corr_ratio_real'], self.num_cat_multi['corr_ratio_synth'], self.num_cat_multi['diff_norm_corr_ratio']