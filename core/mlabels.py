class MLabels:
    def __init__(self, data_type):
        self.data_type = data_type
        self.labels = []
        self.avg_loss = self._set_labels('avg_loss', data_type)
        self.avg_prob = self._set_labels('avg_prob', data_type)
        self.avg_accuracy = self._set_labels('avg_accuracy', data_type)

    def _set_labels(self, metric, data_type):
        label = metric + '/' + data_type if data_type else metric
        self.labels.append(label)
        return label


class RecoveryMLabels(MLabels):
    pass
