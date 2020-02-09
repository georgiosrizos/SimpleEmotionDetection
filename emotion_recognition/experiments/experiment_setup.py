import os
import inspect

import numpy as np
import tensorflow as tf

import emotion_recognition.experiments.core as core
import emotion_recognition.data_provider as data_provider
from emotion_recognition.common import dict_to_struct, make_dirs_safe

# TODO: Number of targets fix.
# TODO: Add global max pool
# TODO: Add fancy attention
# TODO: Multiple domains.


def train(configuration):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = repr(configuration.GPU)

    ####################################################################################################################
    # Interpret configuration arguments.
    ####################################################################################################################
    dropout_keep_prob_eff = 0.5

    train_steps_per_epoch =  configuration.train_sample_no // configuration.train_batch_size
    if configuration.train_sample_no % configuration.train_batch_size > 0:
        train_steps_per_epoch += 1
    print(train_steps_per_epoch)
    train_split_name_list = ["train", ]

    valid_steps_per_epoch = configuration.valid_sample_no // configuration.valid_batch_size
    if configuration.valid_sample_no % configuration.valid_batch_size > 0:
        valid_steps_per_epoch += 1
    print(valid_steps_per_epoch)
    valid_split_name_list = ["valid", ]

    test_steps_per_epoch = configuration.test_sample_no // configuration.test_batch_size
    if configuration.test_sample_no % configuration.test_batch_size > 0:
        test_steps_per_epoch += 1
    print(test_steps_per_epoch)
    test_split_name_list = ["test", ]

    # train_steps_per_epoch = 5
    # train_split_name_list = ["train", ]
    #
    # valid_steps_per_epoch = 5
    # valid_split_name_list = ["valid", ]
    #
    # test_steps_per_epoch = 5
    # test_split_name_list = ["test", ]

    if configuration.task == "single":
        targets = ["emotion"]
        number_of_classes = [3, ]
    elif configuration.task == "multi":
        targets = ["emotion",
                   "arousal",
                   "valence"]
        number_of_classes = [3, 3, 3]
    else:
        raise ValueError("Invalid task type.")
    number_of_targets = sum(number_of_classes)

    file_path_suffix = configuration.framework + "_" + \
                       configuration.task + "_" + \
                       "global_max_pooling" + "_" + \
                       repr(configuration.trial)

    data_folder = configuration.target_folder

    tfrecords_dir = configuration.data_folder + "/tf_records"
    train_dir = data_folder + "/ckpt/train/" + file_path_suffix
    checkpoint_dir = train_dir
    log_dir = data_folder + "/ckpt/log/" + file_path_suffix
    test_pred_dir = log_dir + "/test_pred"
    results_log_file = data_folder + "/losses/" + file_path_suffix

    model_path_last = log_dir + "/last"
    model_path_best = log_dir + "/best"

    model_trained = data_folder + "/ckpt/log/" + \
                    configuration.framework + "_" + \
                       configuration.task + "_" + \
                       "global_max_pooling" + "_" + \
                    repr(configuration.trial) + "/"

    starting_epoch = 0
    starting_best_performance = - 1.0

    ####################################################################################################################
    # Form computational graph.
    ####################################################################################################################
    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            ############################################################################################################
            # Get dataset iterators.
            ############################################################################################################
            dataset_train = data_provider.get_split(tfrecords_dir,
                                                    is_training=True,
                                                    split_name_list=train_split_name_list,
                                                    batch_size=configuration.train_batch_size,
                                                    seq_length=configuration.seq_length,
                                                    buffer_size=train_steps_per_epoch + 1)
            dataset_valid = data_provider.get_split(tfrecords_dir,
                                                    is_training=False,
                                                    split_name_list=valid_split_name_list,
                                                    batch_size=configuration.valid_batch_size,
                                                    seq_length=configuration.seq_length,
                                                    buffer_size=valid_steps_per_epoch + 1)
            dataset_test = data_provider.get_split(tfrecords_dir,
                                                   is_training=False,
                                                   split_name_list=test_split_name_list,
                                                   batch_size=configuration.test_batch_size,
                                                   seq_length=configuration.seq_length,
                                                   buffer_size=test_steps_per_epoch + 1)

            iterator_train = tf.data.Iterator.from_structure(dataset_train.output_types,
                                                             dataset_train.output_shapes)
            iterator_valid = tf.data.Iterator.from_structure(dataset_valid.output_types,
                                                             dataset_valid.output_shapes)
            iterator_test = tf.data.Iterator.from_structure(dataset_test.output_types,
                                                            dataset_test.output_shapes)

            next_element_train = iterator_train.get_next()
            next_element_valid = iterator_valid.get_next()
            next_element_test = iterator_test.get_next()

            init_op_train = iterator_train.make_initializer(dataset_train)
            init_op_valid = iterator_valid.make_initializer(dataset_valid)
            init_op_test = iterator_test.make_initializer(dataset_test)

            ############################################################################################################
            # Define placeholders.
            ############################################################################################################
            batch_size_tensor = tf.placeholder(tf.int32)
            # seq_length_tensor = tf.placeholder(tf.int32)

            subject_ids_train = tf.placeholder(tf.int32, (None, ))
            audio_train = tf.placeholder(tf.float32, (None, configuration.seq_length))
            emotion_train = tf.placeholder(tf.float32, (None, 3))
            arousal_train = tf.placeholder(tf.float32, (None, 3))
            valence_train = tf.placeholder(tf.float32, (None, 3))
            dominance_train = tf.placeholder(tf.float32, (None, 3))
            pre_pad_length_train = tf.placeholder(tf.int32, (None, ))

            subject_ids_test = tf.placeholder(tf.int32, (None, ))
            audio_test = tf.placeholder(tf.float32, (None, configuration.seq_length))
            emotion_test = tf.placeholder(tf.float32, (None, 3))
            arousal_test = tf.placeholder(tf.float32, (None, 3))
            valence_test = tf.placeholder(tf.float32, (None, 3))
            dominance_test = tf.placeholder(tf.float32, (None, 3))
            pre_pad_length_test = tf.placeholder(tf.int32, (None, ))

            ############################################################################################################
            # Other placeholders.
            ############################################################################################################
            # is_training_dropout_tensor = tf.placeholder(tf.bool, shape=[])
            # is_training_batchnorm_tensor = tf.placeholder(tf.bool, shape=[])

            ############################################################################################################
            # Define model graph and get model.
            ############################################################################################################
            # Select model framework.
            if configuration.framework == "end2end":
                get_model_framework = core.get_end2end_model
            else:
                raise ValueError("Invalid framework.")
            with tf.variable_scope("Model"):
                pred_mean_train = get_model_framework()(audio_train,
                                                        batch_size_tensor,
                                                        num_layers=configuration.num_layers,
                                                        hidden_units=configuration.hidden_units,
                                                        use_attention=configuration.use_attention,
                                                        number_of_outputs=number_of_targets,
                                                        batch_size=batch_size_tensor)

            with tf.variable_scope("Model", reuse=True):
                pred_mean_test = get_model_framework()(audio_test,
                                                       batch_size_tensor,
                                                       num_layers=configuration.num_layers,
                                                       hidden_units=configuration.hidden_units,
                                                       use_attention=configuration.use_attention,
                                                       number_of_outputs=number_of_targets,
                                                       batch_size=batch_size_tensor)

            loss_function = core.get_loss_function(configuration.task)
            loss_function_argnames = inspect.getargspec(loss_function)[0]

            tensor_shape_train = [batch_size_tensor, 1]
            flattened_size_train = tensor_shape_train[0] * tensor_shape_train[1]

            single_pred_mean_train = core.flatten_data(pred_mean_train,
                                                       flattened_size_train * number_of_targets)

            tensor_shape_test = [batch_size_tensor, 1]
            flattened_size_test = tensor_shape_test[0] * tensor_shape_test[1]

            single_pred_mean_test = core.flatten_data(pred_mean_test,
                                                      flattened_size_test * number_of_targets)

            loss_kwargs = dict()
            loss_kwargs["pred"] = single_pred_mean_train
            if configuration.task == "single":
                loss_kwargs["true"] = core.flatten_data(emotion_train, flattened_size_train * number_of_targets)
            elif configuration.task == "multi":
                loss_kwargs["true"] = core.flatten_data(tf.concat([emotion_train,
                                                                   arousal_train,
                                                                   valence_train],
                                                                  axis=1),
                                                        flattened_size_train * number_of_targets)
            else:
                raise ValueError("Invalid task definition.")
            loss_kwargs["batch_size"] = configuration.train_batch_size
            loss_kwargs["seq_length"] = configuration.seq_length
            loss_kwargs["flattened_size"] = flattened_size_train
            loss_kwargs = {kw: loss_kwargs[kw] for kw in loss_function_argnames}

            loss = loss_function(**loss_kwargs)

            vars = tf.trainable_variables()
            model_vars = [v for v in vars if v.name.startswith("Model")]
            vars = [v for v in vars]

            saver = tf.train.Saver({v.name: v for v in vars if v.name.startswith("Model")})

            total_loss = tf.reduce_sum(loss)
            optimizer = tf.train.AdamOptimizer(configuration.initial_learning_rate).minimize(total_loss,
                                                                                             var_list=vars)
            ############################################################################################################
            # Initialize variables and perform experiment.
            ############################################################################################################
            sess.run(tf.global_variables_initializer())

            ############################################################################################################
            # Train base model.
            ############################################################################################################
            print("Start training base model.")
            print("Fresh base model.")
            make_dirs_safe(data_folder + "/losses/")
            losses_fp = open(results_log_file + ".txt", "w+")
            measure_names = ["epoch",
                             "valid-accuracy", "test-accuracy",
                             "valid-macro-f1", "test-macro-f1",
                             "valid-au-roc-micro", "test-au-roc-micro",
                             "valid-au-roc-macro", "test-au-roc-macro",
                             "valid-au-prc-micro", "test-au-prc-micro",
                             "valid-au-prc-macro","test-au-prc-macro"]
            losses_fp.write("\t".join(measure_names) + "\n")
            losses_fp.close()
            for ee, epoch in enumerate(range(starting_epoch, configuration.num_epochs + starting_epoch)):
                print("Train Base model.")

                config_epoch_pass = dict()
                config_epoch_pass["sess"] = sess
                config_epoch_pass["init_op"] = init_op_train
                config_epoch_pass["steps_per_epoch"] = train_steps_per_epoch
                config_epoch_pass["next_element"] = next_element_train
                config_epoch_pass["batch_size"] = configuration.train_batch_size
                config_epoch_pass["seq_length"] = configuration.seq_length
                config_epoch_pass["input_gaussian_noise"] = configuration.input_gaussian_noise
                config_epoch_pass["targets"] = targets
                config_epoch_pass["get_vars"] = [(optimizer, "no", None),
                                                 (total_loss, "loss", "loss"),
                                                 (pred_mean_train, "yes", "pred")]
                config_epoch_pass["feed_dict"] = {batch_size_tensor: "batch_size",
                                                      audio_train: "audio",
                                                      emotion_train: "emotion",
                                                      arousal_train: "arousal",
                                                      valence_train: "valence",
                                                      dominance_train: "dominance",
                                                  }

                config_epoch_pass["saver"] = {model_path_last: saver}

                config_epoch_pass = dict_to_struct(config_epoch_pass)

                train_items = core.run_epoch(config_epoch_pass)

                if ee == 0:
                    best_performance = starting_best_performance

                if (ee+1) % configuration.val_every_n_epoch == 0:
                    print("Valid Base model.")
                    config_epoch_pass = dict()
                    config_epoch_pass["sess"] = sess
                    config_epoch_pass["init_op"] = init_op_valid
                    config_epoch_pass["steps_per_epoch"] = valid_steps_per_epoch
                    config_epoch_pass["next_element"] = next_element_valid
                    config_epoch_pass["batch_size"] = configuration.valid_batch_size
                    config_epoch_pass["seq_length"] = configuration.seq_length
                    config_epoch_pass["input_gaussian_noise"] = configuration.input_gaussian_noise
                    config_epoch_pass["targets"] = targets
                    config_epoch_pass["get_vars"] = [(pred_mean_test, "yes", "pred")]
                    config_epoch_pass["feed_dict"] = {batch_size_tensor: "batch_size",
                                                      audio_test: "audio",
                                                      emotion_test: "emotion",
                                                      arousal_test: "arousal",
                                                      valence_test: "valence",
                                                      dominance_test: "dominance",
                                                      }

                    config_epoch_pass["saver"] = {model_path_last: saver}

                    config_epoch_pass = dict_to_struct(config_epoch_pass)

                    valid_items = core.run_epoch(config_epoch_pass)

                    # valid_pred = valid_items.pred
                    # valid_true = valid_items.true

                    print(valid_items.emotion.pred.shape)
                    print(valid_items.emotion.true.shape)

                    # valid_accuracy = core.single_task_multi_class_accuracy_np(valid_items.emotion.pred, valid_items.emotion.true)
                    valid_all_measures = core.single_task_get_all_measures(valid_items.emotion.pred, valid_items.emotion.true)
                    valid_current_performance = valid_all_measures["au-prc-macro"]

                    print(epoch, valid_current_performance)

                    if valid_current_performance > best_performance:
                        best_performance = valid_current_performance

                        saver.save(sess, model_path_best)
                else:
                    print(epoch)

            saver.restore(sess, model_path_best)

            losses_fp.close()
            print("Test Base model.")
            config_epoch_pass = dict()
            config_epoch_pass["sess"] = sess
            config_epoch_pass["init_op"] = init_op_test
            config_epoch_pass["steps_per_epoch"] = test_steps_per_epoch
            config_epoch_pass["next_element"] = next_element_test
            config_epoch_pass["batch_size"] = configuration.test_batch_size
            config_epoch_pass["seq_length"] = configuration.seq_length
            config_epoch_pass["input_gaussian_noise"] = configuration.input_gaussian_noise
            config_epoch_pass["targets"] = targets
            config_epoch_pass["get_vars"] = [(pred_mean_test, "yes_mc", "pred")]
            config_epoch_pass["feed_dict"] = {batch_size_tensor: "batch_size",
                                              audio_test: "audio",
                                              emotion_test: "emotion",
                                              arousal_test: "arousal",
                                              valence_test: "valence",
                                              dominance_test: "dominance", }

            config_epoch_pass["saver"] = None

            config_epoch_pass = dict_to_struct(config_epoch_pass)

            test_items = core.run_epoch(config_epoch_pass)

            # test_accuracy = core.single_task_multi_class_accuracy_np(test_items.emotion.pred, test_items.emotion.true)
            test_all_measures = core.single_task_get_all_measures(test_items.emotion.pred, test_items.emotion.true)
            test_current_performance = test_all_measures["au-prc-macro"]

            print(test_current_performance)

            append_to_results_file(results_log_file,
                                   epoch,
                                   valid_all_measures,
                                   test_all_measures)
            np.save(model_trained + "valid_true", valid_items.emotion.true)
            np.save(model_trained + "valid_pred", valid_items.emotion.pred)
            np.save(model_trained + "test_true", test_items.emotion.true)
            np.save(model_trained + "test_pred", test_items.emotion.pred)


def append_to_results_file(results_log_file, epoch, valid_current_performance, test_current_performance):
    measure_names = ["accuracy", "macro-f1", "au-roc-micro", "au-roc-macro", "au-prc-micro", "au-prc-macro"]
    measure_list = list()
    for n in measure_names:
        measure_list.append(valid_current_performance[n])
        measure_list.append(test_current_performance[n])
    measure_list = [repr(v) for v in measure_list]

    losses_fp = open(results_log_file + ".txt", "a+")
    losses_fp.write(repr(epoch) + "\t" + "\t".join(measure_list) + "\n")
    losses_fp.close()
