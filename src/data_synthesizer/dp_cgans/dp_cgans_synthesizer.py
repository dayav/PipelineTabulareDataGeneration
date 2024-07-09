import pandas as pd
import json
from dp_cgans import DP_CGAN

from IPython.utils.capture import capture_output


class DP_CGAN_SYNTH() :

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, 
                 rounding='auto', min_value='auto', max_value='auto', private=False) :
        self._synth = DP_CGAN( field_names, field_types, field_transformers,
                 anonymize_fields, primary_key, constraints, table_metadata,
                 embedding_dim, generator_dim, discriminator_dim,
                 generator_lr, generator_decay, discriminator_lr,
                 discriminator_decay, batch_size, discriminator_steps,
                 log_frequency, verbose, epochs, pac, cuda, 
                 rounding, min_value, max_value, private)
        
        self._training_report = dict()
        self._training_report['field_names'] = field_names
        self._training_report['field_types'] = field_types
        self._training_report['field_transformers'] = field_transformers
        self._training_report['anonymize_fields'] = anonymize_fields
        self._training_report['primary_key'] = primary_key
        self._training_report['constraints'] = constraints
        self._training_report['table_metadata'] = table_metadata
        self._training_report['embedding_dim'] = embedding_dim
        self._training_report['generator_dim'] = generator_dim
        self._training_report['discriminator_dim'] = discriminator_dim
        self._training_report['generator_lr'] = generator_lr
        self._training_report['generator_decay'] = generator_decay
        self._training_report['discriminator_lr'] = discriminator_lr
        self._training_report['discriminator_decay'] = discriminator_decay
        self._training_report['discriminator_decay'] = discriminator_decay
        self._training_report['discriminator_decay'] = discriminator_decay
        self._training_report['batch_size'] = batch_size
        self._training_report['discriminator_steps'] = discriminator_steps
        self._training_report['log_frequency'] = log_frequency
        self._training_report['verbose'] = verbose
        self._training_report['nb_epochs'] = epochs
        self._training_report['pac'] = pac
        self._training_report['cuda'] = cuda
        self._training_report['rounding'] = rounding
        self._training_report['min_value'] = min_value
        self._training_report['max_value'] = max_value
        self._training_report['private'] = private       


    def fit(self, data):
        self._synth.fit(data)

        self._create_training_output()

    def _create_training_output(self) :
        loss = pd.read_csv('loss_output_{}.txt'.format(self._training_report['nb_epochs']), delimiter=',', header=None)
        self._training_report['Epoch'] = [i for i in range(loss.shape[0])]
        self._training_report['Generator Loss'] = [float(i[8:]) for i in loss[1]]
        self._training_report['Discriminator Loss'] = [float(i[7:]) for i in loss[2]]

    def save_training_report_to_json(self, path) :
        with open(path, "w") as fp:
            json.dump(self._training_report, fp, indent=2)
    