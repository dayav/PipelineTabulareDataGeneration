

import gc
from joblib import Parallel, delayed
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors  
from sklearn.preprocessing import MinMaxScaler


def compare_rows(i, df_real, df_synth, categorical_cols, numeric_cols, col_length, weights_feature):
            
    comp_results = pd.DataFrame(index=[i for i in range(len(df_real))], columns=df_real.columns)
    
    for col in categorical_cols:
        comp_results[col] = df_real[col] == df_synth.iloc[i][col]
        
    for col in numeric_cols:
        comp_results[col] = abs(df_real[col] - df_synth.iloc[i][col])
    
    
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

def dissimilarity(weights_feature, read_data , synth_data) :

    # df_synth = df_real_cardio_ctgan_first
    # df_real = df_real_cardio_train
    categorical_cols = read_data.select_dtypes(include='category').columns
    numeric_cols = read_data.select_dtypes(exclude='category').columns

    col_length = len(synth_data.columns)
    df_length = len(synth_data)

    min_diss_gen_idx = []
    min_diss_gen = []
    min_diss = []
    comp_results_df = []
    last_i = 0
    first_index = 0
    last_length = 5000

    while first_index != df_length :
       
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

def normalize_to_sum_1(numbers):
    total = sum(numbers)
    normalized_numbers = [x / total for x in numbers]
    return normalized_numbers

def sampling_reject(model, real_data, min_diss, cat_features, nb_data_sampled ) :
     
    synth_data = model.sample(nb_data_sampled)
    
    for column in cat_features :
        real_data[column] = real_data[column].astype('category')
        synth_data[column] = synth_data[column].astype('category')
    
    dcr_list = []
    weigths = [1] * len(real_data.columns)
    min_results_tests_dfirst_new, _ = dissimilarity(normalize_to_sum_1(weigths), real_data, synth_data)
    min_results_dict = {i : minimum   for i, minimum in zip(range(len(min_results_tests_dfirst_new)), min_results_tests_dfirst_new)}

    min_results_dict_filt = [i for i in min_results_dict if min_results_dict[i] < min_diss ]

    i = 0
    not_finished = True
    while not_finished :
        data_candidate = model.sample(1)
        weigths = [1] * len(real_data.columns)
        categorical_cols = real_data.select_dtypes(include='category').columns
        numeric_cols = real_data.select_dtypes(exclude='category').columns
        min_results_tests_candidate_new, minimum_candidate_real = compare_rows(0,real_data, data_candidate, categorical_cols, numeric_cols,  24, normalize_to_sum_1(weigths))
        if (min_results_tests_candidate_new > min_diss) :
            synth_data = synth_data.drop(min_results_dict_filt[i])
            # Concatenate DataFrames
            synth_data = pd.concat([synth_data, data_candidate])
            i += 1
            if i == len(min_results_dict_filt) :
                not_finished = False
            for column in cat_features :
                synth_data[column] = synth_data[column].astype('category')

    return synth_data.reset_index(drop=True)


def sampling_reject_epsilon(model, real_data, min_epsilon, cat_features, num_features, nb_data_sampled ) :

    synth_data = model.sample(nb_data_sampled)

    real_data_dummies = pd.get_dummies(real_data, columns=cat_features, dtype=int)
    synth_data_dummies = pd.get_dummies(synth_data, columns=cat_features, dtype=int)
    real_data_dummies_scaled = real_data_dummies.copy()
    synth_data_dummies_scaled = synth_data_dummies.copy()

    extra_columns = set(real_data_dummies_scaled.columns) - set(synth_data_dummies_scaled.columns)

    for column in extra_columns:
        synth_data_dummies_scaled[column] = 0

    synth_data_dummies_scaled = synth_data_dummies_scaled[real_data_dummies_scaled.columns]

    scaler = MinMaxScaler() 
    real_data_dummies_scaled[num_features] = scaler.fit_transform(real_data_dummies_scaled[num_features])
    synth_data_dummies_scaled[num_features] = scaler.transform(synth_data_dummies_scaled[num_features])

    nbrs = NearestNeighbors(n_neighbors = 2).fit(real_data_dummies_scaled)
    distance, indice = nbrs.kneighbors(real_data_dummies_scaled)

    nbrs_hat = NearestNeighbors(n_neighbors = 1).fit(real_data_dummies_scaled)
    distance_hat, indice_hat = nbrs_hat.kneighbors(synth_data_dummies_scaled)

    R_Diff= distance_hat[:,0] - distance[:,1]

    epsilon_id = np.sum(R_Diff<0) / nb_data_sampled

    while (epsilon_id >= min_epsilon):
        data_candidate = model.sample(1)
        data_candidate_dummies = pd.get_dummies(data_candidate, columns=cat_features, dtype=int)
        data_candidate_dummies[num_features] = scaler.transform(data_candidate_dummies[num_features])
        extra_columns = set(real_data_dummies_scaled.columns) - set(data_candidate_dummies.columns)
        for column in extra_columns:
            data_candidate_dummies[column] = 0
        
        data_candidate_dummies = data_candidate_dummies[real_data_dummies_scaled.columns]
        distance_temp, _ = nbrs_hat.kneighbors(data_candidate_dummies)
        should_changhe = distance_hat[:,0] < distance_temp
        if (should_changhe.any()) :

            argmin_distance = distance_hat[:,0].argmin()
            synth_data.iloc[argmin_distance] = data_candidate.iloc[0]
            distance_hat[:,0][argmin_distance] = distance_temp[:,0][0]
            
            R_Diff= distance_hat[:,0] - distance[:,1]
            epsilon_id = np.sum(R_Diff<0) / nb_data_sampled

    return synth_data.reset_index(drop=True)

def get_epsilon(real_data, synth_data, cat_features, num_features) :
    real_data_dummies = pd.get_dummies(real_data, columns=cat_features, dtype=int)
    synth_data_dummies = pd.get_dummies(synth_data, columns=cat_features, dtype=int)
    real_data_dummies_scaled = real_data_dummies.copy()
    synth_data_dummies_scaled = synth_data_dummies.copy()

    extra_columns = set(real_data_dummies_scaled.columns) - set(synth_data_dummies_scaled.columns)

    for column in extra_columns:
        synth_data_dummies_scaled[column] = 0

    synth_data_dummies_scaled = synth_data_dummies_scaled[real_data_dummies_scaled.columns]

    scaler = MinMaxScaler() 
    real_data_dummies_scaled[num_features] = scaler.fit_transform(real_data_dummies_scaled[num_features])
    synth_data_dummies_scaled[num_features] = scaler.transform(synth_data_dummies_scaled[num_features])

    nbrs = NearestNeighbors(n_neighbors = 2).fit(real_data_dummies_scaled)
    distance, indice = nbrs.kneighbors(real_data_dummies_scaled)

    nbrs_hat = NearestNeighbors(n_neighbors = 1).fit(real_data_dummies_scaled)
    distance_hat, indice_hat = nbrs_hat.kneighbors(synth_data_dummies_scaled)

    R_Diff= distance_hat[:,0] - distance[:,1]

    return np.sum(R_Diff<0) / len(synth_data)