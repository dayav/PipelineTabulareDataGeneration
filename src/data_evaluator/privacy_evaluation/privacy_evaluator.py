import sys
import pickle
import numpy as np
import pandas as pd
from math import sqrt
from ..base_evaluator import BaseEvaluator
from data_loader import DataLoader
from .mia_stdg import evaluate_membership_attack
from scipy.spatial import distance
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from scipy.stats import truncnorm

import gower
import gc
import prince

from statistics import mean
from enum import Enum

class SimilarityType(Enum):
    EUCLIDEAN = 0 
    HAUSDORFF = 1 
    COSINE = 2 
    MAHALANOBIS = 3
    GOWER = 4
    DISSIMILARITY = 5
    DISSIMILARITY_CATEGORICAL = 6
    DISSIMILARITY_NUMERICAL = 7
    DISSIMILARITY_MEAN_IMBALANCED = 8
    DISSIMILARITY_WITH_PREPROCESSING_CAT = 9
    EPSILON_DISSIMILARITY = 10

class PrivacyEvaluator(BaseEvaluator) :

    def __init__(self, real_train, synth, real_test, qid_columns, non_quid_columns):
        """
        Initialize the PrivacyEvaluator.

        Parameters:
        - real_train: The real training data.
        - synth: The synthetic data.
        - real_test: The real testing data.
        """
        super().__init__(real_train, synth)
        self._real_test = real_test
        self._data_synthetic_qid = None
        self._data_synthetic_risk = None
        self._data_real_qid = None
        self._data_real_risk = None
        self._attacker_qid_data = None
        self._attacker_non_qid_data = None
        self.attributes_synthetic_models = {}
        self.attributes_real_models = {}
        self._all_qid_columns = qid_columns
        self._non_quid_columns = non_quid_columns

        self._prepare_data(qid_columns)


    def _get_concatenate_data(self):

        backup_real = self._real.copy(deep=True)
        backup_synth = self._synth.copy(deep=True)

        backup_real['Label'] = np.zeros(self._real.shape[0]).astype('int8')
        backup_synth['Label'] = np.ones(self._synth.shape[0])

        #mix real and synthetic records

        frames = [backup_real, backup_synth]

        #return the concatenate dataframe with the mixed samples
        data = pd.concat(frames).sample(frac=1)

        last_ix = len(data.columns) - 1
        return data.drop(['Label'], axis=1), data[['Label']]

    def closest_distance_to_record_stdg(self, proportions, thresholds) :
        real_concat = pd.concat([self._real, self._real_test])
        train_data_indexes = self._real['ID'].tolist()

        precision_values = dict()
        accuracy_values = dict()
        
        for threshold in thresholds :
            precision_values[threshold] = []
            accuracy_values[threshold] = []
            for proportion in proportions :
                attacker_data = real_concat.sample(frac = proportion)

                precision, accuracy = evaluate_membership_attack(attacker_data, train_data_indexes, self._synth, threshold)
                precision_values[threshold].append(precision)
                accuracy_values[threshold].append(accuracy)
        return precision_values, accuracy_values
    
    def _prepare_data(self, qid_columns):
        """Sample and deduplicate data for the attacker."""

        self._qid_columns = qid_columns

        real_data = self._real.drop_duplicates(subset=self._qid_columns, keep='first')
        synth_data = self._synth.drop_duplicates(subset=self._qid_columns, keep='first')
        test_data = self._real_test.drop_duplicates(subset=self._qid_columns, keep='first')

        self._data_synthetic_qid = synth_data[self._qid_columns] 
        self._data_synthetic_risk = synth_data[self._non_quid_columns]
        self._data_real_qid = real_data[self._qid_columns]
        self._data_real_risk = real_data[self._non_quid_columns]

        self._attacker_qid_data = test_data[self._qid_columns]
        self._attacker_non_qid_data = test_data[self._non_quid_columns]


    def _train_attributes_prediction_models(self, isSyntheticData):
        """
        Train models to predict non-QID attributes based on QID attributes.

        Parameters:
        - categorical_columns_qid: Categorical columns among the QIDs.
        - numerical_columns_qid: Numerical columns among the QIDs.
        - non_quid_columns: Columns that are not QIDs (potentially sensitive attributes).
        """

        if (isSyntheticData) :
            data_quid = self._data_synthetic_qid
            data_risk = self._data_synthetic_risk
            attributes_models = self.attributes_synthetic_models
        else :
            data_quid = self._data_real_qid
            data_risk = self._data_real_risk
            attributes_models = self.attributes_real_models

        categorical_columns_qid = [x for x in self._qid_columns if x in self._categorical_columns]
        numerical_columns_qid = [x for x in self._qid_columns if x in self._numerical_columns]

        categorical_idx = [data_quid.columns.get_loc(cat_col) for cat_col in categorical_columns_qid]
        numerical_idx = [data_quid.columns.get_loc(num_col) for num_col in numerical_columns_qid]

        transformers = [
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_idx),
            ('num', StandardScaler(), numerical_idx)
        ]
        col_transform = ColumnTransformer(transformers=transformers)

        # for attribute in tqdm(self._non_quid_columns, desc="Training Models", unit="attribute"):
        for attribute in self._non_quid_columns :
            if attribute in self._categorical_columns:
                model = DecisionTreeClassifier(random_state=64)
            else:
                model = DecisionTreeRegressor(random_state=64)
            
            pipeline = Pipeline(steps=[('prep', col_transform), ('m', model)])
            pipeline.fit(data_quid, data_risk[attribute])
            attributes_models[attribute] = pipeline

    def train_attributes_synthetic_prediction_models(self):
        self._train_attributes_prediction_models(True)

    def train_attributes_real_prediction_models(self):
        self._train_attributes_prediction_models(False)

    def predict_attribute(self, isSynthetic) :
        # Dictionary to store predictions
        predictions = {}
        attributes_models = self.attributes_synthetic_models if isSynthetic else self.attributes_real_models
        # for attribute in tqdm(self._non_quid_columns, desc="Evaluating Predictions", unit="attribute"):
        for attribute in self._non_quid_columns :
            results = attributes_models[attribute].predict(self._attacker_qid_data)
            predictions[attribute] = results

        return predictions

    def _evaluate_attribute_prediction(self, isSynthetic):
        
        self._prepare_data(self._all_qid_columns)


        if isSynthetic :
            self.train_attributes_synthetic_prediction_models()
        else : 
            self.train_attributes_real_prediction_models()
        
        predictions_dict = self.predict_attribute(isSynthetic)
        results = {}
        results_std = {}

        for attribute in predictions_dict :
            pred = []
            if attribute in self._categorical_columns :
                result = accuracy_score(self._attacker_non_qid_data[attribute] , predictions_dict[attribute])
                
                for true_label, prediction in  zip(self._attacker_non_qid_data[attribute], predictions_dict[attribute]) :
                    accuracy = accuracy_score([true_label], [prediction])
                    bool_r = True if accuracy == 1 else False
                    pred.append(bool_r)
                results_std_attribute = pred
            else :
                result = sqrt(mean_squared_error(self._attacker_non_qid_data[attribute] , predictions_dict[attribute]))
                for true_label, prediction in  zip(self._attacker_non_qid_data[attribute], predictions_dict[attribute]) :
                    accuracy = sqrt(mean_squared_error([true_label], [prediction]))
                    bool_r = True if (accuracy < 0.06 )  else False
                    pred.append(bool_r)
                results_std_attribute = pred
            results[attribute] = result
            results_std[attribute] =  np.round(np.sum(results_std_attribute)/len(results_std_attribute),2)
        
        return results_std
    
    def pairwise_euclidean_distance_stdg(self, scaled_real: pd.DataFrame, scaled_synth: pd.DataFrame) :
        # from stdg evaluation metric
        sum_distances = 0
        sum_sq_distances = 0
        n = 0

        for i in range(scaled_synth.shape[0]):
            # Compute distances for one row at a time
            distances_row = distance.cdist(scaled_synth[i:i+1], scaled_real, 'euclidean')[0]
            sum_distances += np.sum(distances_row)
            sum_sq_distances += np.sum(distances_row**2)
            n += len(distances_row)

        mean_distance = sum_distances / n
        std_distance = np.sqrt(sum_sq_distances/n - mean_distance**2)
        return np.round(mean_distance,4), np.round(std_distance,4)
    
    def hausdorff_distance_stdg(self, scaled_real: pd.DataFrame, scaled_synth: pd.DataFrame) :
        # from stdg evaluation metric
        hausdorff_dist = distance.directed_hausdorff(scaled_synth, scaled_real)[0]
        return  np.round(hausdorff_dist,4)
    
    def rts_similarity_stdg(self, scaled_real: pd.DataFrame, scaled_synth: pd.DataFrame) :
        # from stdg evaluation metric
        str_sim = cosine_similarity(scaled_synth, scaled_real)
        return  str_sim
     
    def mahalanobis_sim(self ) :
                    # define the data preparation for the columns
        categorical_idx = [self._real.columns.get_loc(cat_col) for cat_col in self._categorical_columns]
        t = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_idx)]
        
        col_transform = ColumnTransformer(transformers=t)
        _real = col_transform.fit_transform(self._real)
        _synth = col_transform.transform(self._synth)

        if isinstance(_real, np.ndarray) :
            data_real = _real
            data_synth = _synth
        
        elif isinstance(_synth, csr_matrix) :
            data_real = _real.toarray()
            data_synth = _synth.toarray()

        mean_vector = np.mean(data_real, axis=0)
        covariance_matrix = np.cov(data_real, rowvar=False)

        # Check condition number
        cond_number = np.linalg.cond(covariance_matrix)

        if cond_number < 1/sys.float_info.epsilon:
            # Safe to invert
            inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
        else:
            # Use pseudoinverse
            inverse_covariance_matrix = np.linalg.pinv(covariance_matrix)

        distances = []
        for row in data_synth :
            distance = self._mahalanobis_distance(row, mean_vector, inverse_covariance_matrix)
            distances.append(distance)

        return distances

    def _mahalanobis_distance(self, x, mean, inv_cov_matrix):
        x_moins_mean = x - mean
        return np.sqrt(np.dot(np.dot(x_moins_mean, inv_cov_matrix), x_moins_mean.T))
    
    def get_gower_matrix_with_famd(self, save_path = None) :
        is_categorical = [col in self._categorical_columns for col in self._real.columns]

        famd = prince.FAMD(
            n_components=5,  # number of components to keep
            n_iter=3,        # number of iterations of the power method
            copy=True,       # whether to copy the data or operate in-place
            check_input=True, # whether to check the consistency of the inputs
            engine='sklearn',    # backend to use, 'auto' uses FBPCA if installed, otherwise 'sklearn'
            random_state=42   # a random state for reproducibility
        )

        columns_to_convert= self._real.select_dtypes(exclude='category').columns
        for column in columns_to_convert:
            self._real[column] = self._real[column].astype(float)
            self._synth[column] = self._synth[column].astype(float)

        # Fit FAMD to your data and transform it
        famd = famd.fit(self._real)       # df is the dataframe containing mixed types
        df_real = famd.transform(self._real)
        df_synth = famd.transform(self._synth)

        real_synth = gower.gower_matrix(df_real, df_synth)
        synth_real = gower.gower_matrix(df_synth, df_real)

        dif_synth_real_u = np.triu(synth_real)
        dif_synth_real_l = np.tril(real_synth, k=-1)

        del real_synth
        del synth_real

        dif_synth_real = dif_synth_real_u + dif_synth_real_l

        del dif_synth_real_u
        del dif_synth_real_l

        if save_path == None :
            return dif_synth_real
        else :
            with open(save_path +'.pkl', 'wb') as file:
                pickle.dump(dif_synth_real, file)
            del dif_synth_real
            return None

    def get_gower_matrix(self, save_path = None) :
        is_categorical = [col in self._categorical_columns for col in self._real.columns]

        real_synth = gower.gower_matrix(self._real, self._synth, cat_features=is_categorical)
        synth_real = gower.gower_matrix(self._synth, self._real, cat_features=is_categorical)

        dif_synth_real_u = np.triu(synth_real)
        dif_synth_real_l = np.tril(real_synth, k=-1)

        del real_synth
        del synth_real

        dif_synth_real = dif_synth_real_u + dif_synth_real_l

        del dif_synth_real_u
        del dif_synth_real_l

        if save_path == None :
            return dif_synth_real
        else :
            with open(save_path +'.pkl', 'wb') as file:
                pickle.dump(dif_synth_real, file)
            del dif_synth_real
            return None

    def get_gower_matrix_by_type(self, save_path = None) :
        is_categorical =[True] * self._categorical_columns
        is_not_categorical = [False] * self._numerical_columns

        real_synth = gower.gower_matrix(self._real[self._categorical_columns], self._synth[self._categorical_columns], cat_features=is_categorical)
        synth_real = gower.gower_matrix(self._synth[self._categorical_columns], self._real[self._categorical_columns], cat_features=is_categorical)

        dif_synth_real_u = np.triu(synth_real)
        dif_synth_real_l = np.tril(real_synth, k=-1)

        del real_synth
        del synth_real

        dif_synth_real_cat = dif_synth_real_u + dif_synth_real_l

        del dif_synth_real_u
        del dif_synth_real_l

        real_synth = gower.gower_matrix(self._real[self._numerical_columns], self._synth[self._numerical_columns], is_not_categorical)
        synth_real = gower.gower_matrix(self._synth[self._numerical_columns], self._real[self._numerical_columns], is_not_categorical)

        dif_synth_real_u = np.triu(synth_real)
        dif_synth_real_l = np.tril(real_synth, k=-1)

        del real_synth
        del synth_real

        dif_synth_real_num = dif_synth_real_u + dif_synth_real_l

        del dif_synth_real_u
        del dif_synth_real_l

        if save_path == None :
            return dif_synth_real_cat, dif_synth_real_num
        else :
            with open(save_path +'.pkl', 'wb') as file:
                pickle.dump(dif_synth_real, file)
            del dif_synth_real
            return None

    def gower_sim(self) :        
        dif_synth_real = self.get_gower_matrix()
        row_mins = np.min(dif_synth_real, axis=1)
        del dif_synth_real

        return row_mins

    def dissimilarity(self, weights_feature, read_data , synth_data, drop_id = False) :

        # df_synth = df_real_cardio_ctgan_first
        # df_real = df_real_cardio_train
        categorical_cols = read_data.select_dtypes(include='category').columns
        numeric_cols = read_data.select_dtypes(exclude='category').columns

        col_length = len(self._real.columns)
        df_length = len(self._real)

        min_diss_gen_idx = []
        min_diss_gen = []
        min_diss = []
        comp_results_df = []
        last_i = 0
        first_index = 0
        last_length = 5000

        while first_index != df_length :

            def compare_rows(i, df_real, df_synth, categorical_cols, numeric_cols, col_length, weights_feature):

                if drop_id : 
                    df_real_used = df_real.drop(i)
                else : 
                    df_real_used = df_real
                comp_results = pd.DataFrame(index=[i for i in range(len(df_real_used))], columns=df_real_used.columns)
                
                for col in categorical_cols:
                    comp_results[col] = df_real_used[col] == df_synth.iloc[i][col]
                    
                for col in numeric_cols:
                    comp_results[col] = abs(df_real_used[col] - df_synth.iloc[i][col])   
                
                
                comp_results = comp_results.dropna()
                comp_results[categorical_cols] = comp_results[categorical_cols].astype(int).replace({1: 0, 0: 1})
                
                for col in numeric_cols:
                    column_range = comp_results[col].max() - comp_results[col].min()
                    if column_range != 0:
                        comp_results[col] = (comp_results[col] - comp_results[col].min()) / column_range
                
                weighted_comp_results = comp_results * weights_feature
                weighted_comp_results['Sum'] = weighted_comp_results.sum(axis=1)
                # weighted_comp_results['Sum norm CTGAN'] = weighted_comp_results['Sum']
                
                min_comp_results = weighted_comp_results['Sum'].min()
                min_comp_results_idx = weighted_comp_results['Sum'].idxmin()
                del comp_results
                del weighted_comp_results
                return min_comp_results, min_comp_results_idx

            
            results = Parallel(n_jobs=-1)(delayed(compare_rows)(i, read_data, synth_data, categorical_cols, numeric_cols, col_length, weights_feature) for i in range(first_index, last_length))
            min_diss_values = []
            min_diss_idx_values = []
            for min_diss, min_diss_idx in results:
                min_diss_values.append(min_diss)
                min_diss_idx_values.append(min_diss_idx)
            min_diss_gen = min_diss_gen + min_diss_values
            min_diss_gen_idx = min_diss_gen_idx + min_diss_idx_values 

            
            del min_diss
            first_index = last_length
            left_length = df_length - last_length
            added_last_length = min(5000, left_length)
            last_length += added_last_length
            gc.collect()
            print(first_index,' ------finished------', last_length)
        return min_diss_gen, min_diss_gen_idx
        
    
    def dissimilarity_all(self) :
        list_ = [1] * len(self._real.columns)
        weights_feature = self.normalize_to_sum_1(list_)
        diss_real, min_diss_gen_idx_real = self.dissimilarity(weights_feature = weights_feature, read_data = self._real, synth_data = self._synth)
        diss_test, min_diss_gen_idx_test = self.dissimilarity(weights_feature = weights_feature, read_data = self._real_test , synth_data = self._synth)
        
        sum_real = sum(diss_real)
        sum_test = sum(diss_test)
        combined_total = sum_real + sum_test
        share_real = (sum_real / combined_total) * 100
        return diss_real, diss_test, min_diss_gen_idx_real, min_diss_gen_idx_test, share_real
    
    def epsilon_dissimilarity(self) :
        list_ = [1] * len(self._real.columns)
        weights_feature = self.normalize_to_sum_1(list_)
        diss_real, min_diss_gen_idx_real = self.dissimilarity(weights_feature = weights_feature, read_data = self._real , synth_data = self._real, drop_id= True)
        diss_synth, min_diss_gen_idx_synth = self.dissimilarity(weights_feature = weights_feature, read_data = self._synth , synth_data = self._real)
    
        compare = np.array(diss_synth) < np.array(diss_real)
        epsilon = np.sum(compare)/ len(self._real)
        return diss_real, diss_synth, min_diss_gen_idx_real, min_diss_gen_idx_synth, epsilon
    
    def dissimilarity_numerical(self) :
        is_numerical = [int(col in self._numerical_columns) for col in self._real.columns]
        weights_feature = self.normalize_to_sum_1(is_numerical)
        return self.dissimilarity(weights_feature = weights_feature, read_data = self._real, synth_data = self._synth)

    def dissimilarity_categorical(self) :
        is_categorical = [int(col in self._categorical_columns) for col in self._real.columns]
        weights_feature = self.normalize_to_sum_1(is_categorical)
        return self.dissimilarity(weights_feature = weights_feature, read_data = self._real, synth_data = self._synth)

    def dissimilarity_mean_imbalanced(self) :
        mean_imbalance_ratios = []
        for column in self._real.columns :
            if column in self._categorical_columns :
                class_counts = self._real[column].value_counts()

                imbalance_ratios = []

                for label, count in class_counts.items():
                    imbalance_ratio = max(class_counts) / count
                    imbalance_ratios.append(imbalance_ratio)

                # Calculate the mean imbalance ratio
                mean_imbalance_ratio = sum(imbalance_ratios) / len(imbalance_ratios)
                mean_imbalance_ratios.append(mean_imbalance_ratio)
            else :
                mean_imbalance_ratios.append(1)

        mean_imbalance_ratios = [1.0 / x for x in mean_imbalance_ratios]
        
        weights_feature = self.normalize_to_sum_1(mean_imbalance_ratios)
        return self.dissimilarity(weights_feature = weights_feature, read_data = self._real, synth_data = self._synth)

    def dissimilarity_with_preprocessing_cat(self) :
        real_modified = self.create_preprocess_data_frame(self._real)
        real_modified = real_modified.select_dtypes(exclude='category')
        synth_modified = self.create_preprocess_data_frame(self._synth)
        synth_modified = synth_modified.select_dtypes(exclude='category')

        list_ = [1] * len(self._real.columns)
        weights_feature = self.normalize_to_sum_1(list_)

        return self.dissimilarity(weights_feature = weights_feature, read_data = real_modified, synth_data = synth_modified)


    
    def create_preprocess_data_frame(self, data_frame):
        data_modified = data_frame.copy(deep=True)

        for cat_feat in self._categorical_columns :
            sum_cum_cat = data_modified[cat_feat].value_counts(normalize=True).sort_values(ascending=False).cumsum()
            sum_cum_cat_dict = sum_cum_cat.to_dict()
            feature_interval = {}
            first = 0
            for cat in sum_cum_cat_dict :
                value = sum_cum_cat_dict[cat]
                feature_interval[cat] = (first, value)
                first = value

            truncated_dists = {}
            for cat, value in feature_interval.items():
                lower_bound = value[0]  # Lower bound in standard deviations
                upper_bound = value[1]
                mean = (upper_bound + lower_bound) / 2
                std_dev = (upper_bound - lower_bound) / 6
                # Convert the bounds to the scale of the distribution
                a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
                truncated_dist = truncnorm(a, b, loc=mean, scale=std_dev)
                truncated_dists[cat] = truncated_dist

            def convert_to_distribution(value) :
                return truncated_dists[value].rvs(1)[0]

            num_feature_added = cat_feat+'_converted'
            data_modified[num_feature_added] = data_modified[cat_feat].apply(convert_to_distribution)
            data_modified[num_feature_added] = data_modified[num_feature_added].astype('float64')

        return data_modified


    def normalize_to_sum_1(self, numbers):
        total = sum(numbers)
        normalized_numbers = [x / total for x in numbers]
        return normalized_numbers
    
    def evaluate_attribute_synthetic_prediction(self) :
        return self._evaluate_attribute_prediction(True)

    def evaluate_attribute_real_prediction(self) :
        return self._evaluate_attribute_prediction(False)
    
    def evaluate_similarity_stdg(self, distance=SimilarityType.EUCLIDEAN) :
        scaled_real = MinMaxScaler().fit_transform(self._real[self._numerical_columns])
        scaled_synth = MinMaxScaler().fit_transform(self._synth[self._numerical_columns])
        if distance==SimilarityType.EUCLIDEAN :
            distances = self.pairwise_euclidean_distance_stdg(scaled_real, scaled_synth)
        elif distance == SimilarityType.HAUSDORFF :
            distances = self.hausdorff_distance_stdg(scaled_real, scaled_synth)
        elif distance == SimilarityType.COSINE :
            distances = self.rts_similarity_stdg(scaled_real, scaled_synth)
        elif distance == SimilarityType.MAHALANOBIS :
            distances = self.mahalanobis_sim()
        elif distance == SimilarityType.GOWER :
            distances = self.gower_sim()
            gc.collect()
        elif distance == SimilarityType.DISSIMILARITY :
            distances = self.dissimilarity_all()
            gc.collect()
        elif distance == SimilarityType.DISSIMILARITY_NUMERICAL : 
            distances = self.dissimilarity_numerical()
            gc.collect()
        elif distance == SimilarityType.DISSIMILARITY_CATEGORICAL : 
            distances = self.dissimilarity_categorical()
            gc.collect()
        elif distance == SimilarityType.DISSIMILARITY_MEAN_IMBALANCED : 
            distances = self.dissimilarity_mean_imbalanced()
            gc.collect()
        elif distance == SimilarityType.DISSIMILARITY_WITH_PREPROCESSING_CAT : 
            distances = self.dissimilarity_with_preprocessing_cat()
            gc.collect()   
        elif distance == SimilarityType.EPSILON_DISSIMILARITY : 
            distances = self.epsilon_dissimilarity()
            gc.collect()  
        return distances

    