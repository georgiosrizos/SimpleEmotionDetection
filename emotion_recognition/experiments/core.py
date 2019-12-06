import numpy as np
import tensorflow as tf
import sklearn

from emotion_recognition.common import dict_to_struct


def flatten_data(data, flattened_size):
    flattened_data = tf.reshape(data[:, :],
                                (-1,))
    flattened_data = tf.reshape(flattened_data,
                                (flattened_size, 1, 1, 1))
    return flattened_data


def get_end2end_model():

    def wrapper(*args, **kwargs):
        return end2end_bilstm_model(end2end_audio_model(*args), **kwargs)

    return wrapper


def single_task_multi_class(pred, true):
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=true,
                                                 logits=pred)

    ce = tf.squeeze(tf.reduce_sum(ce))
    return ce


def stable_softmax(X):
    exps = np.exp(X - np.max(X, 1).reshape((X.shape[0], 1)))
    return exps / np.sum(exps, 1).reshape((X.shape[0], 1))


def single_task_multi_class_np(pred, true):
    """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector.
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    true = true.argmax(axis=1)

    m = true.shape[0]
    p = stable_softmax(pred)
    # p = spspec.softmax(pred, 1)
    # We use multidimensional array indexing to extract
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m), true])
    loss = np.sum(log_likelihood) / m
    return loss


def single_task_multi_class_accuracy_np(pred, true):
    pred = stable_softmax(pred)
    # pred = spspec.softmax(pred, 1)
    pred = pred.argmax(axis=1)
    true = true.argmax(axis=1)

    acc = sklearn.metrics.accuracy_score(true, pred, normalize=True, sample_weight=None)

    return acc


def single_task_get_all_measures(pred_logits, true_indicator):
    measures = dict()

    # Accuracy.
    pred_prob = stable_softmax(pred_logits)
    # pred_labels_indices = pred_prob.argmax(axis=1)
    # pred_labels = np.zeros_like(pred_prob)
    # pred_labels[:, pred_labels_indices] = 1.0

    pred_labels = (pred_prob.argmax(1)[:,None] == np.arange(pred_prob.shape[1])).astype(int)

    # true_indicator = np.around(true_indicator).astype(dtype=np.int32)
    # pred_labels = np.around(pred_labels).astype(dtype=np.int32)

    # print(true_indicator.sum() - true_indicator.shape[0])
    # print(pred_labels.sum() - pred_labels.shape[0])

    # print(true_indicator.dtype)
    # print(pred_logits.dtype)
    # print(pred_labels.dtype)

    acc = sklearn.metrics.accuracy_score(true_indicator, pred_labels, normalize=True, sample_weight=None)
    measures["accuracy"] = acc

    # Macro-F1.
    macro_f1 = sklearn.metrics.f1_score(true_indicator, pred_labels, average="macro")
    measures["macro-f1"] = macro_f1

    # AU-ROC.
    au_roc_macro = sklearn.metrics.roc_auc_score(true_indicator, pred_prob, average="macro")
    au_roc_micro = sklearn.metrics.roc_auc_score(true_indicator, pred_prob, average="micro")
    measures["au-roc-macro"] = au_roc_macro
    measures["au-roc-micro"] = au_roc_micro

    # AU-PRC
    au_prc_macro = sklearn.metrics.average_precision_score(true_indicator, pred_prob, average="macro")
    au_prc_micro = sklearn.metrics.average_precision_score(true_indicator, pred_prob, average="micro")
    measures["au-prc-macro"] = au_prc_macro
    measures["au-prc-micro"] = au_prc_micro

    return measures


def get_loss_function(task):
    if task == "single":
        loss_function = single_task_multi_class
    elif task == "multi":
        loss_function = single_task_multi_class
    else:
        raise ValueError("Invalid task type.")

    return loss_function


def end2end_audio_model(audio_frames,
                        batch_size):
    _, seq_length = audio_frames.get_shape().as_list()
    audio_input = tf.reshape(audio_frames, [batch_size, 1 * seq_length, 1])

    net = tf.layers.conv1d(audio_input, 64, 8, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling1d(net, 10, 10)

    net = tf.layers.conv1d(net, 128, 6, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling1d(net, 8, 8)

    net = tf.layers.conv1d(net, 256, 6, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling1d(net, 8, 8)

    net = tf.reshape(net, [batch_size, seq_length // 640, 256])
    return net


def end2end_lstm_model(net,
                       num_layers,
                       hidden_units,
                       number_of_outputs,
                       use_attention,
                       batch_size):
    _, seq_length, num_features = net.get_shape().as_list()

    def _get_cell(l_no):
        lstm = tf.contrib.rnn.LSTMCell(hidden_units,
                                       use_peepholes=True,
                                       cell_clip=100,
                                       state_is_tuple=True)
        # if dropout_keep_prob < 1.0:
        #     if l_no == 0:
        #         lstm = tf.contrib.rnn.DropoutWrapper(lstm,
        #                                              input_keep_prob=dropout_keep_prob,
        #                                              output_keep_prob=dropout_keep_prob,
        #                                              state_keep_prob=dropout_keep_prob,
        #                                              variational_recurrent=True,
        #                                              input_size=num_features,
        #                                              dtype=tf.float32)
        #     else:
        #         lstm = tf.contrib.rnn.DropoutWrapper(lstm,
        #                                              output_keep_prob=dropout_keep_prob,
        #                                              state_keep_prob=dropout_keep_prob,
        #                                              variational_recurrent=True,
        #                                              input_size=hidden_units,
        #                                              dtype=tf.float32)
        return lstm

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([_get_cell(l_no) for l_no in range(num_layers)], state_is_tuple=True)

    outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)
    # outputs, _ = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_fw, stacked_lstm_bw, net, dtype=tf.float32)
    #
    # outputs = outputs[0] + outputs[1]

    if seq_length is None:
        seq_length = -1

    net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
    attention = tf.layers.dense(net, 1)
    attention = tf.reshape(attention, (batch_size, seq_length, 1))
    attention = tf.nn.softmax(attention, axis=1)
    net = tf.reshape(outputs, (batch_size, seq_length, hidden_units))
    net = tf.multiply(net, attention)
    net = tf.reduce_sum(net, axis=1)

    net = tf.reshape(net, (batch_size, hidden_units))

    mean_prediction = tf.layers.dense(net, number_of_outputs)
    mean_prediction = tf.reshape(mean_prediction, (batch_size, number_of_outputs))

    return mean_prediction


def end2end_bilstm_model(net,
                         num_layers,
                         hidden_units,
                         use_attention,
                         number_of_outputs,
                         batch_size):
    _, seq_length, num_features = net.get_shape().as_list()

    def _get_cell(l_no):
        lstm = tf.contrib.rnn.LSTMCell(hidden_units,
                                       use_peepholes=True,
                                       cell_clip=100,
                                       state_is_tuple=True)
        # if dropout_keep_prob < 1.0:
        #     if l_no == 0:
        #         lstm = tf.contrib.rnn.DropoutWrapper(lstm,
        #                                              input_keep_prob=dropout_keep_prob,
        #                                              output_keep_prob=dropout_keep_prob,
        #                                              state_keep_prob=dropout_keep_prob,
        #                                              variational_recurrent=True,
        #                                              input_size=num_features,
        #                                              dtype=tf.float32)
        #     else:
        #         lstm = tf.contrib.rnn.DropoutWrapper(lstm,
        #                                              output_keep_prob=dropout_keep_prob,
        #                                              state_keep_prob=dropout_keep_prob,
        #                                              variational_recurrent=True,
        #                                              input_size=hidden_units,
        #                                              dtype=tf.float32)

        # if l_no == (num_layers - 1):
        #     lstm = tf.contrib.rnn.AttentionCellWrapper(lstm,
        #                                                50)

        return lstm

    stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell([_get_cell(l_no) for l_no in range(num_layers)], state_is_tuple=True)
    stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell([_get_cell(l_no) for l_no in range(num_layers)], state_is_tuple=True)

    # outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_fw, stacked_lstm_bw, net, dtype=tf.float32)

    outputs = outputs[0] + outputs[1]

    if seq_length is None:
        seq_length = -1

    if use_attention:
        # Attention.
        attention = tf.layers.dense(outputs, 1)
        attention = tf.nn.softmax(attention, axis=1)
        outputs = tf.reshape(outputs, (-1, seq_length, hidden_units))  # config.batch_size
        outputs = tf.multiply(outputs, attention)
        outputs = tf.reduce_sum(outputs, axis=1)

        mean_prediction = tf.reshape(outputs, (-1, hidden_units))  # config.batch_size
        mean_prediction = tf.layers.dense(mean_prediction, number_of_outputs)

        mean_prediction = tf.reshape(mean_prediction, (batch_size, number_of_outputs))
    else:
        # net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
        #
        # mean_prediction = tf.layers.dense(net, number_of_outputs)
        # mean_prediction = tf.reshape(mean_prediction, (batch_size, seq_length, number_of_outputs))
        # mean_prediction = tf.reshape(mean_prediction[:, -1, :], (batch_size, number_of_outputs))

        net = tf.reshape(outputs, (batch_size, seq_length, hidden_units))
        net = tf.layers.max_pooling1d(net, seq_length, seq_length)

        mean_prediction = tf.layers.dense(net, number_of_outputs)
        mean_prediction = tf.reshape(mean_prediction, (batch_size, number_of_outputs))

    return mean_prediction


def run_epoch(config_struct):
    sess = config_struct.sess
    init_op = config_struct.init_op
    steps_per_epoch = config_struct.steps_per_epoch
    next_element = config_struct.next_element
    batch_size = config_struct.batch_size
    seq_length = config_struct.seq_length
    input_gaussian_noise = config_struct.input_gaussian_noise
    get_vars = config_struct.get_vars
    input_feed_dict = config_struct.feed_dict
    targets = config_struct.targets
    saver = config_struct.saver

    mc_samples = 1

    if len(targets) == 1:
        number_of_targets = 3
    else:
        number_of_targets = 11

    out_tf = list()
    track_var = list()
    var_names = list()
    counter = 0
    for t in get_vars:
        out_tf.append(t[0])
        track_var.append(t[1])
        var_names.append(t[2])
        counter += 1

    batch_size_sum = 0

    # Initialize an iterator over the dataset split.
    sess.run(init_op)

    # Store variable sequence.
    stored_variables = dict()
    for target in targets:
        stored_variables[target] = dict()
        stored_variables[target]["true"] = np.empty((steps_per_epoch * batch_size, number_of_targets),
                                                     dtype=np.float32)

    for i, track in enumerate(track_var):
        if track in ["yes", "yes_mc"]:
            for target in targets:
                stored_variables[target][var_names[i]] = np.empty((steps_per_epoch * batch_size, number_of_targets),
                                                                   dtype=np.float32)

        if track == "yes_mc":
            for target in targets:
                stored_variables[target][var_names[i] + "_epi"] = np.empty((steps_per_epoch * batch_size, number_of_targets),
                                                                            dtype=np.float32)
        if track == "loss":
            stored_variables[var_names[i]] = list()

    temp_mc = dict()
    temp = dict()
    for i, track in enumerate(track_var):
        if track in ["yes", "yes_mc"]:
            temp_mc[var_names[i]] = np.empty((batch_size,
                                                 number_of_targets,
                                                 mc_samples),
                                                dtype=np.float32)
            temp[var_names[i]] = None
        if track == "yes_mc":
            temp[var_names[i] + "_epi"] = None

    for step in range(steps_per_epoch):
        batch_tuple = sess.run(next_element)
        # sample_id = batch_tuple["sample_id"]
        # subject_id = batch_tuple["subject_id"]

        emotion = batch_tuple["emotion"]
        arousal = batch_tuple["arousal"]
        valence = batch_tuple["valence"]
        dominance = batch_tuple["dominance"]
        audio = batch_tuple["raw_audio"]

        batch_size_sum += audio.shape[0]

        seq_pos_start = step * batch_size
        seq_pos_end = seq_pos_start + audio.shape[0]

        # # Augment data.
        # jitter = np.random.normal(scale=input_gaussian_noise,
        #                           size=audio.shape)
        # audio_plus_jitter = audio + jitter

        feed_dict = {k: v for k, v in input_feed_dict.items()}
        feed_dict = replace_dict_value(feed_dict, "batch_size", audio.shape[0])
        # feed_dict = replace_dict_value(feed_dict, "audio", audio_plus_jitter)
        feed_dict = replace_dict_value(feed_dict, "audio", audio)
        feed_dict = replace_dict_value(feed_dict, "emotion", emotion)
        feed_dict = replace_dict_value(feed_dict, "arousal", arousal)
        feed_dict = replace_dict_value(feed_dict, "valence", valence)
        feed_dict = replace_dict_value(feed_dict, "dominance", dominance)

        for t in range(mc_samples):
            out_np = sess.run(out_tf,
                              feed_dict=feed_dict)
            for i, track in enumerate(track_var):
                if track in ["yes", "yes_mc"]:
                    temp_mc[var_names[i]][:audio.shape[0], :, t] = out_np[i]

        for i, track in enumerate(track_var):
            if track in ["yes", "yes_mc"]:
                temp[var_names[i]] = np.mean(temp_mc[var_names[i]], axis=2)
            if track == "yes_mc":
                if mc_samples > 1:
                    temp[var_names[i] + "_epi"] = np.var(temp_mc[var_names[i]], axis=2)
                else:
                    temp[var_names[i] + "_epi"] = temp_mc[var_names[i]].reshape((batch_size,
                                                  number_of_targets))

        stored_variables["emotion"]["true"][seq_pos_start:seq_pos_end, :] = np.squeeze(emotion)
        if number_of_targets > 3:
            stored_variables["arousal"]["true"][seq_pos_start:seq_pos_end, :] = np.squeeze(arousal)
            stored_variables["valence"]["true"][seq_pos_start:seq_pos_end, :] = np.squeeze(valence)
            stored_variables["dominance"]["true"][seq_pos_start:seq_pos_end, :] = np.squeeze(dominance)

        for i, track in enumerate(track_var):
            if track in ["yes", "yes_mc"]:
                for e, target in enumerate(targets):
                    stored_variables[target][var_names[i]][seq_pos_start:seq_pos_end, :] = temp[var_names[i]][:audio.shape[0], :]
            if track == "yes_mc":
                for e, target in enumerate(targets):
                    stored_variables[target][var_names[i] + "_epi"][seq_pos_start:seq_pos_end, :] = temp[var_names[i] + "_epi"][:audio.shape[0], :]
            if track == "loss":
                stored_variables[var_names[i]].append(out_np[i])

    for i, track in enumerate(track_var):
        if track == "loss":
            stored_variables[var_names[i]] = np.mean(np.array(stored_variables[var_names[i]]))

    if saver is not None:
        for path, s in saver.items():
            s.save(sess, path)

    for target in targets:
        stored_variables[target]["true"] = stored_variables[target]["true"][:batch_size_sum, :]
        stored_variables[target]["pred"] = stored_variables[target]["pred"][:batch_size_sum, :]
        # stored_variables[target]["true"] = np.dstack(
        #     [stored_variables[target]["true"].reshape(list(stored_variables[target]["true"].shape)) for target in
        #      targets])[:batch_size_sum, :, :]
        # stored_variables[target]["pred"] = np.dstack(
        #     [stored_variables[target]["pred"].reshape(list(stored_variables[target]["pred"].shape)) for target in
        #      targets])[:batch_size_sum, :, :]
        stored_variables[target] = dict_to_struct(stored_variables[target])

    stored_variables = dict_to_struct(stored_variables)

    return stored_variables


def replace_dict_value(input_dict, old_value, new_value):
    for k, v in input_dict.items():
        if isinstance(v, str):
            if v == old_value:
                input_dict[k] = np.nan_to_num(new_value, copy=True)
    return input_dict
