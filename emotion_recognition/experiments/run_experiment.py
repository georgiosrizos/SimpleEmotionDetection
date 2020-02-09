from emotion_recognition.experiments.experiment_setup import train
from emotion_recognition.common import dict_to_struct

for t in range(0, 1):
    # time.sleep(4 * 60 * 60)

    # Make the arguments' dictionary.
    configuration = dict()
    configuration["data_folder"] = "/data/Data/data_folder/preprocessed_data/IEMOCAP/self_support"
    configuration["target_folder"] = "/data/Data/Experiments/quality/self_support"
    configuration["trial"] = t

    configuration["framework"] = "end2end"  # ["deepspectrum", "end2end", "zixing"]
    configuration["task"] = "single"  # ["single", "multi"]

    configuration["input_gaussian_noise"] = 0.1
    configuration["num_layers"] = 2
    configuration["hidden_units"] = 256
    configuration["use_attention"] = False
    configuration["initial_learning_rate"] = 0.00001
    configuration["seq_length"] = 510720
    configuration["train_sample_no"] = 1689
    configuration["train_aug_sample_no"] = 4452
    configuration["valid_sample_no"] = 535
    configuration["test_sample_no"] = 558
    configuration["train_batch_size"] = 10
    configuration["valid_batch_size"] = 10
    configuration["test_batch_size"] = 10
    # configuration["train_size"] = 137
    # configuration["valid_size"] = 60
    # configuration["test_size"] = 73
    configuration["num_epochs"] = 40
    configuration["val_every_n_epoch"] = 1

    configuration["GPU"] = 1

    configuration = dict_to_struct(configuration)

    train(configuration)

