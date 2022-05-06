
class IDataGenerator:

    def __init__(self, dataset, input_groups, target_groups):
        self._dataset       = dataset
        self._input_groups  = input_groups
        self._target_groups = target_groups

    @property
    def dataset(self):
        return self._dataset

    @property
    def input_groups(self):
        return self._input_groups

    @property
    def target_groups(self):
        return self._target_groups

    @property
    def weights(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """Get batch with index `index`.

        Returns a batch constructed from `self.data_loader` with index `index`.

        Parameters
        ----------
        index : int
            Batch index. 0 <= `index` < len(self)

        Returns
        -------
        inputs : dict
            Dictionary of batches of input variables where keys are input
            labels: [ 'input_slice', 'input_png3d', 'input_png2d' ] and values
            are the batches themselves.
            If self.vars_input_* is None then the corresponding input will be
            missing from `inputs`.
        targets : dict
            Dictionary of batches of target variables where keys are target
            labels: [ 'target_total', 'target_primary' ] and values are the
            batches themselves.
            If self.var_target_* is None then the corresponding targets will be
            missing from `targets`.
        weight : list of ndarray
            List of weights for each target is `targets`.
        """

        raise NotImplementedError

