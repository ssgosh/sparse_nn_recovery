def _set_labels(metric, data_type):
    return metric + '/' + data_type if data_type else metric

class MLabels:
    def __init__(self, data_type):
        self.data_type = data_type
        self.avg_loss = _set_labels('avg_loss', data_type)
        self.avg_prob = _set_labels('avg_prob', data_type)
        self.avg_accuracy = _set_labels('avg_accuracy', data_type)


class RecoveryMLabels(MLabels):
    pass
