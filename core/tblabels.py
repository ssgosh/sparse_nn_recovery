# TODO: Divide each label below into aggregate and per-class
class TBLabels:
    """
    Top-level labels shown in Tensorboard

    NOTE: This is the nomenclature:
      Per-class:                Stats for each class separately
      Aggregate:                Average over all classes
      Per-Batch:                Stats for one training batch only, reported every batch.
      Per-Epoch:                Stats reported every epoch.
      In case of per-epoch,
      _{train/validation}_{i}:         Adversarial Stats for ith intermittent (adversarial) {train/validation} dataset
      _{train/validation}:            Adversarial Stats averaged over all intermittent datasets; Real data stats
    """
    ########### Per-batch Stats. ##################
    # These are the ones we'll look at to see if our model is even training. #
    PER_BATCH_ADV_TRAINING_AGGREGATE = "yyy_adversarial_training_per_batch_stats_aggregate"
    PER_BATCH_ADV_TRAINING_PER_CLASS = "zzz_adversarial_training_per_batch_stats_per_class"

    ################## Per-epoch Stats ###################################
    # These are the ones that we'll look at to make decisions. #

    # On sample of 5k real training data points, and combined adversarial training data (1k samples each) from all past epochs
    PER_EPOCH_ADV_TRAINING_AGGREGATE_TRAIN = "adversarial_training_per_epoch_stats_aggregate_train"

    # On adversarial training dataset generated in epoch i
    @staticmethod
    def PER_EPOCH_ADV_TRAINING_AGGREGATE_TRAIN(i): return f"yyy_adversarial_training_per_epoch_stats_aggregate_train_{i}"

    # On real validation dataset and combined adversarial validation data from all past epochs
    PER_EPOCH_ADV_VALIDATIONING_AGGREGATE_VALIDATION = "adversarial_training_per_epoch_stats_aggregate_validation"

    # On intermittent adversarial validation datasets
    @staticmethod
    def PER_EPOCH_ADV_TRAINING_AGGREGATE_VALIDATION(i): return f"yyy_adversarial_training_per_epoch_stats_aggregate_validation_{i}"

    # On external adversarial validation datasets, such as one-pixel attack, Differential Evolution, our held-out network B
    @staticmethod
    def PER_EPOCH_ADV_TRAINING_AGGREGATE_EXTERNAL_VALIDATION(name): return f"adversarial_training_per_epoch_stats_aggregate_external_validation_{name}"

    ####### PER-CLASS versions of above stats ##############

    # Per-epoch. These are the ones that we'll look at to make decisions.
    # On sample of 5k real training data points, and combined adversarial training data (1k samples each) from all past epochs
    PER_EPOCH_ADV_TRAINING_PER_CLASS_TRAIN = "zzz_adversarial_training_per_epoch_stats_per_class_train"

    # On adversarial training dataset generated in epoch i
    @staticmethod
    def PER_EPOCH_ADV_TRAINING_PER_CLASS_TRAIN(i): return f"zzz_adversarial_training_per_epoch_stats_per_class_train_{i}"

    # On real validation dataset and combined adversarial validation data from all past epochs
    PER_EPOCH_ADV_VALIDATIONING_PER_CLASS_VALIDATION = "zzz_adversarial_training_per_epoch_stats_per_class_validation"

    # On intermittent adversarial validation datasets
    @staticmethod
    def PER_EPOCH_ADV_TRAINING_PER_CLASS_VALIDATION(i): return f"zzz_adversarial_training_per_epoch_stats_per_class_validation_{i}"

    # On external adversarial validation datasets, such as one-pixel attack, Differential Evolution, our held-out network B
    @staticmethod
    def PER_EPOCH_ADV_TRAINING_PER_CLASS_EXTERNAL_VALIDATION(name): return f"zzz_adversarial_training_per_epoch_stats_per_class_external_validation_{name}"

    ####### Recovery stats ########

    # Internal ones are logged per recovery batch.
    RECOVERY_INTERNAL_AGGREGATE = "zzz_recovery_internal_aggregate"
    RECOVERY_INTERNAL_PER_CLASS = "zzz_recovery_internal_per_class"

    # recovery_epoch is logged once every epoch on a sample of 100 recovered images
    RECOVERY_EPOCH_AGGREGATE = "recovery_epoch_aggregate"
    RECOVERY_EPOCH_PER_CLASS = "recovery_epoch_per_class"

