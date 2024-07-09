from typing import TypedDict, Optional
import pandas as pd
import copy

from data_loader.data_loader import DataLoader
from data_evaluator.univariate_evaluator import UnivariateEvaluator
from data_synthesizer.pipeline.base_pipeline import Task, TaskType
from data_synthesizer.pipeline.pipeline_results import GenerationResults, PipelineResults
from data_synthesizer.privacy_sampling import get_epsilon, sampling_reject_epsilon


class GenerationTask(Task):
    """Task for generating synthetic data."""
    def __init__(self, model, train_data : pd.DataFrame , cat_features : list, num_features : list):
        super().__init__(model, train_data, cat_features, num_features)


    def process(self, results : PipelineResults) :
        """Process generation by fitting the model and sampling synthetic data."""
        print("Generation processing:")
        self.model.fit(self.train_data)
        self.synth_data = self.model.sample(len(self.train_data))
        generation_results = GenerationResults(synthetic_data =  self.synth_data, generator_model = self.model)

        results['generation_results'] = generation_results
    
class FineTuningGenerationTask(GenerationTask):
    """Task for fine-tuning generation based on mode collapse."""
        
    def process(self, results : PipelineResults):
        """Fine-tune generation by adjusting rare values and resampling."""
        print("Fine Tuning Generation Task")

        self.synth_data = results['generation_results']['synthetic_data']
        self.model = results['generation_results']['generator_model']

        train_data_to_evaluate = DataLoader(dataset=self.train_data).get_dataframe(self.cat_features)
        synth_data_to_evaluate = DataLoader(dataset=self.synth_data).get_dataframe(self.cat_features)

        self._uni_evaluators = UnivariateEvaluator(train_data_to_evaluate, synth_data_to_evaluate)
        mode_collapse = self._uni_evaluators.get_mode_collapse()
        if mode_collapse:
            while mode_collapse:
                print('mode collapse correction')
                mask = pd.Series([False] * len(self.train_data))
            
                for feature, values in mode_collapse.items():
                    print(values)
                    mask |= self.train_data[feature].isin(values)

                filtered_D = self.train_data[mask].reset_index(drop=True)
                print('rare from train : ', len(filtered_D))
                deep_copied_model= copy.deepcopy(self.model)
                deep_copied_model.fit_with_only_rare(filtered_D)
                synth_data_new = deep_copied_model.sample(len(self.train_data))

                most_frequent = {col: self.synth_data[col].value_counts().idxmax() for col in self.synth_data.columns}
                df_combined = self.synth_data.copy()

                for feature, values in mode_collapse.items():
                    for value in values:
                        filter_mask = (synth_data_new[feature] == value)
                        print('filter_mask : ',  filter_mask.any())
                        filtered_df = synth_data_new[filter_mask]
                        print('len filtered_df :', len(filtered_df))
                        sampled_df = filtered_df.sample(n=10, random_state=42) if len(filtered_df) >= 10 else filtered_df
                        df_combined = pd.concat([df_combined, sampled_df], ignore_index=True)

                        most_common_value = most_frequent[feature]
                        filter_most_common = (df_combined[feature] == most_common_value)
                        random_indexes = df_combined[filter_most_common].sample(len(sampled_df)).index
                        df_combined.drop(random_indexes, inplace=True)
                        print('df_combined :', len(df_combined[df_combined[feature] == value]))


                self.synth_data = df_combined.reset_index(drop=True)
                synth_data_to_evaluate = DataLoader(dataset=self.synth_data).get_dataframe(self.cat_features)
                self._uni_evaluators = UnivariateEvaluator(train_data_to_evaluate, synth_data_to_evaluate)
                mode_collapse = self._uni_evaluators.get_mode_collapse()

            self.generation_results = GenerationResults(synthetic_data =  self.synth_data, generator_model = self.model)

            results['generation_results'] = self.generation_results
        else :
            print('No Mode Collpase')
     
class SamplingAndRejectTask(GenerationTask):
    """Task for generating data with rejection sampling based on privacy constraints."""
    def __init__(self, model, train_data : pd.DataFrame , cat_features : list, num_features : list, epsilon: float):
        super().__init__(model, train_data, cat_features, num_features)
        self.epsilon = epsilon
        

    def process(self, results : PipelineResults) -> pd.DataFrame:
        """Process data using epsilon-based rejection sampling."""
        print("Sampling and reject task")

        self.synth_data = results['generation_results']['synthetic_data']
        
        self._epsilon_identifiability = get_epsilon(self.train_data, self.synth_data,
                                                        self.cat_features, self.num_features)
        
        print('initial epsilon : ', self._epsilon_identifiability)
        if self._epsilon_identifiability > self.epsilon:
                
            self.synth_data = sampling_reject_epsilon(self.model, self.train_data, self.epsilon,
                                                    self.cat_features, self.num_features, len(self.train_data))
            
            self.generation_results = GenerationResults(synthetic_data =  self.synth_data)

            results['generation_results'] = self.generation_results