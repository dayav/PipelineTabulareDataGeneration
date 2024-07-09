import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, wasserstein_distance, chi2_contingency, wilcoxon
from scipy.stats import kstest, ks_2samp 
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from tqdm import tqdm
from .base_evaluator import BaseEvaluator

class UnivariateEvaluator(BaseEvaluator) :

    def __init__(self, real, synth) :
        self._real = real
        self._synth = synth
        self._categorical_columns = real.select_dtypes(include='category').columns
        self._numerical_columns = real.select_dtypes(exclude='category').columns

        self.cat_uni = None
        self.num_uni = None
    
    def style_negative(v, props=''):
        return props if v < 0.05 else None
    
    def style_binary(v, props=''):
        return props if v > 0.5 else None


    def cosine_distances(real, synthetic) :

        num_cols = real._get_numeric_data().columns

        cos_s = []

        for c in num_cols :
            cos_s.append(cosine(real[c].values, synthetic[c].values))

        return cos_s
    
    def wass_distances(real, synthetic) :

        num_cols = real._get_numeric_data().columns

        wass_s = []

        for c in num_cols :
            wass_s.append(wasserstein_distance(real[c].values, synthetic[c].values))
        return wass_s

    def mean_comparaison(self):
        mean_compare_dict = dict()

        mean_compare_dict['real_mean'] = self._real[self._numerical_columns].mean()
        mean_compare_dict['synth_mean'] = self._synth[self._numerical_columns].mean()
        mean_compare_dict['abs_diff'] = np.absolute(mean_compare_dict['real_mean'] - mean_compare_dict['synth_mean'])

        mean_compare_dict = pd.DataFrame(data=mean_compare_dict, index=(self._numerical_columns))           
        
        return mean_compare_dict
    
    def std_comparaison(self):
        std_compare_dict = dict()

        std_compare_dict['real_std'] = self._real[self._numerical_columns].std()
        std_compare_dict['synth_std'] = self._synth[self._numerical_columns].std()
        std_compare_dict['abs_diff'] = np.absolute(std_compare_dict['real_std'] - std_compare_dict['synth_std'])

        std_compare_dict = pd.DataFrame(data=std_compare_dict, index=(self._numerical_columns))           
        
        return std_compare_dict
    
    def student_Ttest(self):
        p_s = []
        for col in self._numerical_columns :
            _, p = ttest_ind(self._real[col], self._synth[col])
            p_s.append(p)
        
        return p_s
    
    def Welchs_Ttest(self):
        p_s = []
        for col in self._numerical_columns :
            _, p = ttest_ind(self._real[col], self._synth[col], equal_var=False)
            p_s.append(p)
        
        return p_s

    def mann_whitney_Utest(self):
        p_s = []
        for col in self._numerical_columns :
            stat, p = mannwhitneyu(self._real[col], self._synth[col])
            p_s.append(p)
        return p_s
    
    def wilcoxon_test(self):
        p_s = []
        for col in self._numerical_columns :
            stat, p = wilcoxon(self._real[col], self._synth[col])
            p_s.append(p)
        return p_s
    
    def ks_test(self):
        ks = []
        ks_2samp_ = []
        ks_2samp_p_value = []
        for col in self._numerical_columns:
            ks.append(kstest(self._real[col],'norm')[0]-kstest(self._synth[col],'norm')[0])
            ks_2 = ks_2samp(self._real[col], self._synth[col])
            ks_2samp_.append(ks_2[0])
            ks_2samp_p_value.append(ks_2[1])
        return ks, ks_2samp_, ks_2samp_p_value
    
    def chisquared_tests(self) :
        p_values = []

        for c in self._categorical_columns :
            
            observed = pd.crosstab(self._real[c], self._synth[c], dropna=True)
            #perform chi-squared test
            stat, p, _, _ = chi2_contingency(observed)
            p_values.append(p)

        return p_values
    
    def get_mode_collapse_values(self) :
         return self.cat_uni['mode_collapse_dict']
    
    def get_mode_collapse(self) :
        mode_collapse_dict = {}
        for c in self._categorical_columns :
            observed = pd.crosstab(self._real[c], self._synth[c], dropna=False)
            
            not_mode_collapse = set(self._synth[c].unique()) == set(self._real[c].unique())
            if  not not_mode_collapse :
                cats = pd.Categorical(self._synth[c], categories=self._real[c].unique())
                cat_freq = cats.value_counts()
                cat_freq[cat_freq == 0].index.to_list()
                mode_collapse_dict[c] = cat_freq[cat_freq == 0].index.to_list()
        return mode_collapse_dict
                
    def kl_divergence_categorical(self):
        kl_divergence = []

        for c in self._categorical_columns:
            real_prob = [i/self._real[c].shape[0] for i in self._real[c].value_counts().sort_index(ascending=True).reindex(self._real[c].unique(), fill_value=1e-12)]
            synthetic_prob = [i/self._synth[c].shape[0] for i in self._synth[c].value_counts().sort_index(ascending=True).reindex(self._real[c].unique(), fill_value=1e-12)]
            m = len(real_prob)
            n = len(synthetic_prob)
            
            kl_divergence.append(sum(rel_entr(real_prob, synthetic_prob)))

        return kl_divergence
    
    def jensen_shanon_divergence_categorical(self):
        js_divergence = []

        for c in self._categorical_columns:
            real_prob = np.array([i/self._real[c].shape[0] for i in self._real[c].value_counts().sort_index(ascending=True).reindex(self._real[c].unique(), fill_value=1e-12)])
            synthetic_prob = np.array([i/self._synth[c].shape[0] for i in self._synth[c].value_counts().sort_index(ascending=True).reindex(self._real[c].unique(), fill_value=1e-12)])
            js_pq = jensenshannon(real_prob, synthetic_prob, base=2)

            js_divergence.append(js_pq)

        return js_divergence
    
    
    def kl_divergence_numerical(self):
        kl_divergence = []

        for c in self._numerical_columns:
            # Define or obtain your two numerical datasets (replace these with your data)
            data1 = self._real[c].values
            data2 = self._synth[c].values

            # Perform kernel density estimation (KDE) for both datasets
            kde1 = gaussian_kde(data1)
            kde2 = gaussian_kde(data2)

            # Define the range over which to evaluate the PDFs 
            x_range = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 1000)

            # Compute the PDFs using KDE for the x_range
            pdf1 = kde1(x_range)
            pdf2 = kde2(x_range)

            epsilon = 1e-10
            pdf1 = np.clip(pdf1, epsilon, None)
            pdf2 = np.clip(pdf2, epsilon, None)
    
            kl_divergence.append(np.sum(rel_entr(pdf1, pdf2)))

        return kl_divergence
    
    def jensen_shanon_divergence_numerical(self):
        js_divergence = []

        for c in self._numerical_columns:
            # Define or obtain your two numerical datasets (replace these with your data)
            data1 = self._real[c].values
            data2 = self._synth[c].values

            # Perform kernel density estimation (KDE) for both datasets
            kde1 = gaussian_kde(data1)
            kde2 = gaussian_kde(data2)

            # Define the range over which to evaluate the PDFs 
            x_range = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 1000)

            # Compute the PDFs using KDE for the x_range
            pdf1 = kde1(x_range)
            pdf2 = kde2(x_range)

            epsilon = 1e-10
            pdf1 = np.clip(pdf1, epsilon, None)
            pdf2 = np.clip(pdf2, epsilon, None)
            
            js_pq = jensenshannon(pdf1, pdf2, base=2)

            js_divergence.append( js_pq)

        return js_divergence
    
    def cohen_s_d(self) :
        cohen_s_d_list = []

        for c in self._numerical_columns:
            # Define or obtain your two numerical datasets (replace these with your data)
            group1 = self._real[c].values
            group2 = self._synth[c].values
            # Calculate means and variances
            mean1 = np.mean(group1)
            mean2 = np.mean(group2)
            variance1 = np.var(group1, ddof=1)  # Use ddof=1 for sample variance
            variance2 = np.var(group2, ddof=1)

            # Calculate pooled standard deviation
            n1 = len(group1)
            n2 = len(group2)
            pooled_std = np.sqrt(((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2))

            # Calculate Cohen's d
            cohen_d = (mean1 - mean2) / pooled_std
            cohen_s_d_list.append(abs(cohen_d))
        return cohen_s_d_list
    
    def evaluate_categorical_stat_evaluation(self) :
        univariate_categorical = dict()
        univariate_categorical['KL_divergence_tests'] = self.kl_divergence_categorical()
        univariate_categorical['JS_divergence'] = self.jensen_shanon_divergence_categorical()

        chi_test = pd.DataFrame(data={'chisquared_tests_p' : self.chisquared_tests()}, index=(self._categorical_columns))
        kl = pd.DataFrame(data={'KL_divergence_tests' : self.kl_divergence_categorical()})
        js = pd.DataFrame(data=univariate_categorical, index=(self._categorical_columns))
        
        self.chi_test = chi_test
        self.jensen_shanon = js
        self.jensen_shanon_data_mean = js.mean()

        self.cat_uni = {'chi_test' : chi_test, 'jensen_shanon' : js, 'jensen_shanon_data_mean': js.mean() }

    def evaluate_mode_collapse_values(self) :
        mode_collapse_dict = {}
        for c in self._categorical_columns :
            
            
            observed = pd.crosstab(self._real[c], self._synth[c], dropna=False)
            
            if observed.shape[0] != observed.shape[1] :
                cats = pd.Categorical(self._synth[c], categories=self._real[c].unique())
                cat_freq = cats.value_counts()
                mode_collapse_dict[c] = '/'.join(map(str, cat_freq[cat_freq == 0].index.to_list()))
        
        if self.cat_uni is None :
            self.cat_uni = {'mode_collapse_dict' : mode_collapse_dict }
        else :
            self.cat_uni['mode_collapse_dict'] = mode_collapse_dict
        
    def evaluate_numerical_stat_evaluation(self) :
        univariate_numeric = dict()
        univariate_numeric['mann_whitney'] = self.mann_whitney_Utest()
        univariate_numeric['wilcoxon'] = self.wilcoxon_test()
        univariate_numeric['student_t'] = self.student_Ttest()
        univariate_numeric['welchs_Ttest'] = self.Welchs_Ttest()
        _, _, univariate_numeric['ks_test_p'] = self.ks_test()

        univariate_num_div = dict()
        univariate_num_div['KL_divergence_tests'] = self.kl_divergence_numerical()
        univariate_num_div['JS_divergence'] = self.jensen_shanon_divergence_numerical()
        univariate_num_div['cohen_s_d'] = self.cohen_s_d()

        
        df = pd.DataFrame(data=univariate_numeric, index=(self._numerical_columns))        
        js = pd.DataFrame(data=univariate_num_div, index=(self._numerical_columns))

        self.univariate_num_s = df
        self.univariate_num_js = js
        self.univariate_num_js_data_mean = js.mean()
        self.num_uni = {'univariate_num_s' : df, 'univariate_num_js' : js, 'univariate_num_js_data_mean': js.mean() }

    def get_numerical_stat_evaluation(self) :
        univariate_num_s = self.num_uni['univariate_num_s'].style.applymap(UnivariateEvaluator.style_negative, props='color:red;')
        univariate_num_js = self.num_uni['univariate_num_js'].style.applymap(UnivariateEvaluator.style_binary, props='color:red;', subset= pd.IndexSlice[:, ['JS_divergence']])
        univariate_num_js_data_mean = self.num_uni['univariate_num_js_data_mean']
        
        return univariate_num_s, univariate_num_js, univariate_num_js_data_mean
    
    def get_categorical_stat_evaluation(self) : 
        chi_test = self.cat_uni['chi_test'].style.applymap(UnivariateEvaluator.style_negative, props='color:red;')
        jensen_shanon = self.cat_uni['jensen_shanon'].style.applymap(UnivariateEvaluator.style_binary, props='color:red;', subset= pd.IndexSlice[:, ['JS_divergence']])
        jensen_shanon_mean = self.cat_uni['jensen_shanon_data_mean']
        
        return chi_test, jensen_shanon , jensen_shanon_mean
