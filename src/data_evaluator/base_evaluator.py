class BaseEvaluator :
    def __init__(self, real, synth ) :
        self._real = real
        self._synth = synth
        self._categorical_columns = real.select_dtypes(include='category').columns
        self._numerical_columns = real.select_dtypes(exclude='category').columns
