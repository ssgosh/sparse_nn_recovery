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
      _{train/test}_{i}:         Adversarial Stats for ith intermittent (adversarial) {train/test} dataset
      _{train/test}:            Adversarial Stats averaged over all intermittent datasets; Real data stats
    """
    # Per-batch. These are the ones we'll look at to see if our model is even training.
    PER_BATCH_ADV_TRAINING_AGGREGATE = "adversarial_training_per_batch_stats_aggregate"
    PER_BATCH_ADV_TRAINING_PER_CLASS = "adversarial_training_per_batch_stats_per_class"

    # Per-epoch. These are the ones that we'll look at to make decisions.
    # On sample of 5k real training data points, and combined adversarial training data (1k samples each) from all past epochs
    PER_EPOCH_ADV_TRAINING_AGGREGATE_TRAIN = "adversarial_training_per_epoch_stats_aggregate_train"

    # On adversarial training dataset generated in epoch i
    @staticmethod
    def PER_EPOCH_ADV_TRAINING_AGGREGATE_TRAIN(i): return f"adversarial_training_per_epoch_stats_aggregate_train_{i}"

    # On combined adversarial training data from all past epochs
    #PER_EPOCH_ADV_TRAINING_AGGREGATE_TRAIN_OVERALL = "adversarial_training_per_epoch_stats_aggregate_train_overall"

    @staticmethod
    def PER_EPOCH_ADV_TRAINING_AGGREGATE_TEST(i): return f"adversarial_training_per_epoch_stats_aggregate_test_{i}"

    @staticmethod
    def PER_EPOCH_ADV_TRAINING_AGGREGATE_EXTERNAL_TEST(name): return f"adversarial_training_per_epoch_stats_aggregate_external_test_{name}"
    PER_EPOCH_ADV_TRAINING_PER_CLASS = "adversarial_training_per_epoch_stats_per_class"

    # Recovery stats. Internal ones are logged per recovery batch.
    RECOVERY_INTERNAL = "zzz_recovery_internal"
    RECOVERY_EPOCH = "recovery_epoch"
