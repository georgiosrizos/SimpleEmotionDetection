from pathlib import Path
import os

import tensorflow as tf
import numpy as np
import librosa

# TODO: Select the IEMOCAP folder.
root_dir = "/data/Data/IEMOCAP_full_release"

emotion_list = ["hap", "sad", "ang"]


def read_stats(file_path):
    stats_dict = dict()
    # Original stats.
    with open(file_path, "r") as fp:
        for row in fp:
            row_split = row.strip().split("\t")
            if row_split[1] not in emotion_list:
                continue
            # print(row_split[0])
            # raise ValueError
            stats_dict[row_split[0]] = dict()
            if row_split[1] == "sad":
                stats_dict[row_split[0]]["emotion"] = np.array([1, 0, 0], dtype=np.float32)
            elif row_split[1] == "ang":
                stats_dict[row_split[0]]["emotion"] = np.array([0, 1, 0], dtype=np.float32)
            elif row_split[1] == "hap":
                stats_dict[row_split[0]]["emotion"] = np.array([0, 0, 1], dtype=np.float32)
            else:
                raise ValueError("Invalid emotion case.")

            if row_split[2] == "neg":
                stats_dict[row_split[0]]["arousal"] = np.array([1, 0, 0], dtype=np.float32)
            elif row_split[2] == "neu":
                stats_dict[row_split[0]]["arousal"] = np.array([0, 1, 0], dtype=np.float32)
            elif row_split[2] == "pos":
                stats_dict[row_split[0]]["arousal"] = np.array([0, 0, 1], dtype=np.float32)
            else:
                raise ValueError("Invalid emotion case.")

            if row_split[3] == "neg":
                stats_dict[row_split[0]]["valence"] = np.array([1, 0, 0], dtype=np.float32)
            elif row_split[3] == "neu":
                stats_dict[row_split[0]]["valence"] = np.array([0, 1, 0], dtype=np.float32)
            elif row_split[3] == "pos":
                stats_dict[row_split[0]]["valence"] = np.array([0, 0, 1], dtype=np.float32)
            else:
                raise ValueError("Invalid emotion case.")

            if row_split[4] == "neg":
                stats_dict[row_split[0]]["dominance"] = np.array([1, 0, 0], dtype=np.float32)
            elif row_split[4] == "neu":
                stats_dict[row_split[0]]["dominance"] = np.array([0, 1, 0], dtype=np.float32)
            elif row_split[4] == "pos":
                stats_dict[row_split[0]]["dominance"] = np.array([0, 0, 1], dtype=np.float32)
            else:
                raise ValueError("Invalid emotion case.")

    return stats_dict


file_stats = read_stats(root_dir + "/stats.txt")


def get_portion_to_id(folder):
    p_t_i = dict()
    p_t_i["train"] = list()
    p_t_i["valid"] = list()
    p_t_i["test"] = list()

    list_of_files = librosa.util.find_files(folder)
    for f in list_of_files:
        if not f[-4:] == ".wav":
            raise ValueError("IEMOCAP only has wavs.")
        split_f = f.split("/")
        session_name = split_f[4]
        relative_file_path = "/".join(split_f[4:])

        if relative_file_path not in file_stats.keys():
            continue

        if session_name in ["Session1",
                            "Session2",
                            "Session3"]:
            if ("sentences" in relative_file_path):  # and ("impro" in relative_file_path):
                p_t_i["train"].append(relative_file_path)
        elif session_name in ["Session4"]:
            if ("sentences" in relative_file_path):  # and ("impro" in relative_file_path):
                p_t_i["valid"].append(relative_file_path)
        elif session_name in ["Session5"]:
            if ("sentences" in relative_file_path):  # and ("impro" in relative_file_path):
                p_t_i["test"].append(relative_file_path)
        else:
            raise ValueError("There are only 5 sessions.")

    p_t_i["train"] = sorted(p_t_i["train"])
    p_t_i["valid"] = sorted(p_t_i["valid"])
    p_t_i["test"] = sorted(p_t_i["test"])

    return p_t_i


portion_to_id = get_portion_to_id(root_dir)


def read_csv(path):
    data = list()
    with open(path, "r") as fp:
        for line in fp:
            clean_line = line.strip().split(",")
            data.append(float(clean_line[1]))
    data = np.array(data, dtype=np.float32)
    return data


def get_true(folder, mapping):
    data = dict()
    data["arousal"] = list()
    data["valence"] = list()

    for i in range(1, 10):
        file_name = mapping["test_" + repr(i)]

        data["arousal"].append(read_csv(folder + "/arousal/" + file_name + ".csv"))
        data["valence"].append(read_csv(folder + "/valence/" + file_name + ".csv"))
    data["arousal"] = np.vstack(data["arousal"])
    data["valence"] = np.vstack(data["valence"])
    # print(data["arousal"].shape)
    # print(data["valence"].shape)

    return data


def get_samples(file_path, portion, audio_mean, audio_std, max_seq_len):
    # clip = VideoFileClip(file_path)
    #
    # subsampled_audio = clip.audio.set_fps(16000)
    #
    # # Get audio, image recordings data.
    # audio_frames = []

    audio = librosa.core.load(root_dir + "/" + file_path, sr=48000, mono=False)

    audio = librosa.resample(audio[0], orig_sr=48000, target_sr=16000)

    # audio1 = (audio - np.mean(audio)) / np.std(audio)
    audio = (audio - audio_mean) / audio_std

    pre_pad_length = max_seq_len - audio.size

    if audio.size < max_seq_len:
        audio = np.concatenate([np.zeros((max_seq_len - audio.size, ), dtype=np.float32), audio], axis=0)

    # print(np.sum(np.abs(audio1 - audio2)))

    step_number = audio.size // 640
    step_number_remainder = audio.size % 640
    if step_number_remainder > 0:
        raise ValueError("This shouldn't be possible because it was corrected earlier.")

    audio_frames = audio

    emotion = file_stats[file_path]["emotion"]
    arousal = file_stats[file_path]["arousal"]
    valence = file_stats[file_path]["valence"]
    dominance = file_stats[file_path]["dominance"]

    return audio_frames,\
           emotion, \
           arousal, \
           valence, \
           dominance, \
           pre_pad_length

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_sample(writer, counter, relative_file_path, portion, audio_mean, audio_std, max_seq_len):
    audio,\
    emotion,\
    arousal,\
    valence,\
    dominance, \
    pre_pad_length = get_samples(relative_file_path, portion, audio_mean, audio_std, max_seq_len)

    emotion_shape = np.array(emotion.shape)
    arousal_shape = np.array(arousal.shape)
    valence_shape = np.array(valence.shape)
    dominance_shape = np.array(dominance.shape)
    raw_audio_shape = np.array(audio.shape)

    example = tf.train.Example(features=tf.train.Features(feature={
        'subject_id': _int_feature(counter),
        'emotion': _bytes_feature(emotion.tobytes()),
        # 'emotion_shape': _bytes_feature(emotion_shape.tobytes()),
        'arousal': _bytes_feature(arousal.tobytes()),
        # 'arousal_shape': _bytes_feature(arousal_shape.tobytes()),
        'valence': _bytes_feature(valence.tobytes()),
        # 'valence_shape': _bytes_feature(valence_shape.tobytes()),
        'dominance': _bytes_feature(dominance.tobytes()),
        # 'dominance_shape': _bytes_feature(dominance_shape.tobytes()),
        'raw_audio': _bytes_feature(audio.tobytes()),
        # 'raw_audio_shape': _bytes_feature(raw_audio_shape.tobytes()),
        'pre_pad_length': _int_feature(pre_pad_length),
    }))

    writer.write(example.SerializeToString())


def calculate_stats(split):
    max_seq_len = 0

    sum_value = 0.0
    number_of_values = 0.0

    for relative_file_path in portion_to_id[split]:
        audio = librosa.core.load(root_dir + "/" + relative_file_path, sr=48000, mono=False)

        audio = librosa.resample(audio[0], orig_sr=48000, target_sr=16000)

        if audio.size > max_seq_len:
            max_seq_len = audio.size

        sum_value += np.sum(audio)
        number_of_values += audio.size
    mean_value = sum_value / number_of_values

    sum_squares_value = 0.0
    for relative_file_path in portion_to_id[split]:
        audio = librosa.core.load(root_dir + "/" + relative_file_path, sr=48000, mono=False)

        audio = librosa.resample(audio[0], orig_sr=48000, target_sr=16000)

        sum_squares_value += np.sum(np.power(audio - mean_value, 2.0))
    standard_deviation = np.sqrt(sum_squares_value / number_of_values)

    return mean_value, standard_deviation, max_seq_len


def main(directory):

    # Calculate stats for train set.
    # audio_mean, audio_std, max_seq_len = calculate_stats("train")
    # Stats from training set - without the augmented data!!!!
    audio_mean = 1.864416018729669e-05
    audio_std = 0.08068214311598118
    max_seq_len = 466079
    # print(max_seq_len)  # 466079
    # audio_mean, audio_std, max_seq_len = calculate_stats("valid")
    # print(max_seq_len)  # 389760
    # audio_mean, audio_std, max_seq_len = calculate_stats("test")
    # print(max_seq_len)  # 510560
    # audio_mean, audio_std = 0.0, 1.0

    print(audio_mean, audio_std, max_seq_len)

    # Max seq len after padding.
    max_seq_len = 510560

    step_number = max_seq_len // 640
    step_number_remainder = max_seq_len % 640
    if step_number_remainder > 0:
        max_seq_len += (640 - step_number_remainder)

    # step_number = max_seq_len // (640 // 4)
    # step_number_remainder = max_seq_len % (640 // 4)
    # if step_number_remainder > 0:
    #     max_seq_len += ((640 // 4) - step_number_remainder)

    for portion in portion_to_id.keys():
        print(portion)

        counter = 0

        for relative_file_path in portion_to_id[portion]:
            writer = tf.python_io.TFRecordWriter(
                (directory / 'tf_records' / portion / '{}.tfrecords'.format(counter)
                ).as_posix())
            serialize_sample(writer, counter, relative_file_path, portion, audio_mean, audio_std, max_seq_len)

            counter += 1


if __name__ == "__main__":
    # max_seq_len = 510560
    # step_number = max_seq_len // 640
    # step_number_remainder = max_seq_len % 640
    # if step_number_remainder > 0:
    #     max_seq_len += (640 - step_number_remainder)
    # print(step_number)
    # print(step_number_remainder)
    # print(max_seq_len)

    aa = get_portion_to_id(root_dir)

    print(len(aa["train"]))
    print(len(aa["valid"]))
    print(len(aa["test"]))

    # TODO: Select your output.
    main(Path("/data/Data/data_folder/preprocessed_data/IEMOCAP/self_support"))
