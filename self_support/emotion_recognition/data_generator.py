from io import BytesIO
from pathlib import Path
import collections
import os

import menpo
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
from scipy.io import arff
from scipy.ndimage.filters import generic_filter
from moviepy.editor import VideoFileClip
from menpo.visualize import print_progress

root_dir = "/data/Data/IEMOCAP_full_release"

emotion_list = ["hap", "sad", "ang"]
# emotion_list = ["sad", "ang"]

# portion_to_id = dict(
#     train = [1, 2, 3, 4, 5, 6, 7, 8, 9],
#     valid = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# )

# MAX_TEST_SET = set(['Ses05F_impro04_F023', 'Ses05M_script01_1_M035', 'Ses05F_impro02_F005',
#          'Ses03M_script03_2_M003', 'Ses04M_script01_1_F013', 'Ses01M_script01_1_F039',
#           'Ses04M_script03_2_F030', 'Ses05F_script03_2_M039', 'Ses01F_impro04_F028',
#           'Ses04F_impro01_M021', 'Ses02M_impro06_F006', 'Ses03M_script03_2_F011',
#           'Ses03M_impro06_F014', 'Ses04F_script01_1_M012', 'Ses03F_script01_3_M042',
#            'Ses03F_script02_2_F005', 'Ses01M_impro01_F008', 'Ses04F_script01_1_M013',
#             'Ses05M_script03_2_M029', 'Ses04M_script01_1_F007', 'Ses05F_impro06_M009',
#              'Ses01M_script03_2_F035', 'Ses04F_script02_2_F034', 'Ses01M_script03_2_F015',
#              'Ses01M_script01_1_F016', 'Ses01F_script03_2_F017', 'Ses05F_impro02_F031',
#               'Ses01M_impro06_F004', 'Ses03M_script01_1_F038', 'Ses03M_script01_3_F007',
#             'Ses01M_script01_3_M041', 'Ses05M_script01_1_F004', 'Ses03F_script01_1_M027',
#                  'Ses03F_script03_2_F040',
#                  'Ses04F_impro08_M023', 'Ses03M_impro02_M004', 'Ses04F_script01_1_F004',
#                  'Ses04M_script03_2_M055', 'Ses02M_impro06_M018', 'Ses03M_script03_2_M007',
#               'Ses05M_script03_2_M008', 'Ses04M_script03_2_M051', 'Ses02M_impro05_M013',
#                'Ses05M_script03_2_M038', 'Ses02M_impro06_M023', 'Ses01M_script02_2_F015',
#                 'Ses03M_impro05b_M025', 'Ses04F_script03_2_F044', 'Ses03F_script03_2_F018',
#                  'Ses03M_script01_1_M012', 'Ses04M_impro01_M024', 'Ses02F_script02_2_F035',
#               'Ses03F_impro02_F031', 'Ses04M_script03_2_F019', 'Ses04F_impro04_F016',
#            'Ses02F_script02_2_F024','Ses01M_impro01_F021', 'Ses02M_script01_1_F035',
#            'Ses03F_script03_2_M001', 'Ses03F_impro02_F034', 'Ses05F_script01_1_M035',
#             'Ses05M_script03_2_M028', 'Ses05F_script03_2_F018', 'Ses02F_script03_2_M043',
#              'Ses01F_script03_2_M021', 'Ses05F_impro06_F004', 'Ses05F_impro04_F022',
#               'Ses05M_impro02_F021', 'Ses03F_impro02_M024', 'Ses04M_script01_1_F035',
#                'Ses02F_script03_2_F034', 'Ses04F_impro02_F006', 'Ses04M_script01_1_M021',
#                 'Ses04F_script03_2_F038', 'Ses02M_script01_1_M003', 'Ses03F_script01_3_M031',
#                  'Ses03M_impro06_F001', 'Ses01M_script03_2_M024', 'Ses05M_script03_2_M040',
#                   'Ses04M_script03_2_F049', 'Ses03M_script03_2_F038', 'Ses01M_script03_2_M042',
#                    'Ses04F_script03_2_F026', 'Ses04F_script01_3_M027', 'Ses01M_impro02_F019',
#                     'Ses04F_script03_2_M031', 'Ses04M_script01_1_F022', 'Ses04M_script02_1_F003',
#                      'Ses02M_script01_1_F033', 'Ses04M_impro06_F014', 'Ses02F_script02_2_F020',
#               'Ses03F_impro06_M015', 'Ses03F_script01_2_M015', 'Ses01M_script03_2_M023',
#                'Ses03F_script01_2_F006', 'Ses05F_script03_2_M016', 'Ses04F_script03_2_F027',
#                         'Ses03M_impro05b_M008', 'Ses05M_script02_2_F031', 'Ses05M_script02_2_F018',
#                          'Ses01M_impro06_M019', 'Ses05M_script01_1_F036', 'Ses05M_script03_2_M023',
#                           'Ses05M_impro06_F009', 'Ses01M_impro02_M021', 'Ses02M_script02_2_F020',
#                            'Ses05F_impro05_F034', 'Ses02F_script03_2_M039', 'Ses04M_impro05_M018',
#             'Ses01F_script03_2_F014', 'Ses01M_impro02_F010', 'Ses04M_script02_1_F010',
#             'Ses02F_script01_1_F036', 'Ses04F_impro06_M003', 'Ses03F_script01_1_M022',
#             'Ses05M_impro06_M020', 'Ses05M_script01_3_F021', 'Ses04F_script02_1_F019',
#             'Ses04F_script02_2_M040', 'Ses05F_script03_2_F037', 'Ses02M_script02_2_M038',
#             'Ses04F_impro01_M008', 'Ses01M_impro02_M019', 'Ses03F_script01_3_M029',
#              'Ses01M_script01_2_F013', 'Ses03M_script03_2_F045', 'Ses03M_impro05b_M026',
#              'Ses04M_script03_2_F050', 'Ses01F_script02_2_F036', 'Ses05F_script01_2_F012',
#               'Ses02M_impro06_M022', 'Ses04F_script01_2_F006', 'Ses03F_impro06_F024',
#               'Ses05M_impro02_M021', 'Ses03F_impro08_M011', 'Ses01M_impro06_M010',
#               'Ses03F_script01_3_M032', 'Ses01M_impro06_M000', 'Ses05M_script03_2_M036',
#               'Ses03F_impro06_F007', 'Ses05F_impro02_M032', 'Ses02F_script02_2_M037',
#               'Ses01M_script03_2_F025', 'Ses05M_script03_2_F035', 'Ses03F_impro06_F015',
#               'Ses03M_impro02_F026', 'Ses04M_script01_3_F022', 'Ses01M_script03_2_F034',
#               'Ses05F_script01_2_F010', 'Ses05F_impro02_M004', 'Ses01F_impro02_F017',
#               'Ses05M_script01_1_M023', 'Ses05F_script01_2_M017', 'Ses02M_script01_2_F003',
#               'Ses04F_script01_1_F029', 'Ses03M_impro05a_M021', 'Ses02F_script03_2_F036',
#               'Ses03M_impro02_F016', 'Ses05M_script03_2_M032', 'Ses04M_script02_1_F002',
#               'Ses04F_script03_2_M037', 'Ses04M_script03_2_F000', 'Ses02F_script01_2_F013',
#               'Ses04F_impro02_M001', 'Ses01F_impro06_F022', 'Ses04F_script01_1_M038',
#               'Ses03M_script01_2_F010', 'Ses02M_impro01_M009', 'Ses02F_script03_2_F029',
#               'Ses01M_impro06_M002', 'Ses01F_script01_1_F037', 'Ses01F_script01_2_F004',
#               'Ses03M_impro02_F025', 'Ses04F_script01_1_F035', 'Ses01M_impro05_M023',
#               'Ses03M_script03_2_M041', 'Ses05M_script02_2_F005', 'Ses04M_script03_2_M038',
#               'Ses04M_impro06_M011', 'Ses01M_impro06_M022', 'Ses04M_script03_2_M027',
#               'Ses01M_script02_1_F014', 'Ses02F_script03_2_F042', 'Ses02M_impro04_F009',
#               'Ses02M_script01_1_M038', 'Ses04M_script03_2_M041', 'Ses01F_impro06_F013',
#               'Ses01M_script01_1_F029', 'Ses04F_script03_2_F037'])

# print(len(MAX_TEST_SET))


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

    # Augmented stats.
    file_names = os.listdir("/data/Data/IEMOCAP_full_release/3-emo-augmented")
    for f in file_names:
        f_clean = f.split("_")

        emotion = f_clean[-1][:-4].split("to")
        orig_emotion = int(emotion[0])
        target_emotion = int(emotion[1])

        relative_path = "3-emo-augmented/" + f

        stats_dict[relative_path] = dict()

        if target_emotion == 1:
            stats_dict[relative_path]["emotion"] = np.array([1, 0, 0], dtype=np.float32)
        elif target_emotion == 0:
            stats_dict[relative_path]["emotion"] = np.array([0, 1, 0], dtype=np.float32)
        elif target_emotion == 2:
            stats_dict[relative_path]["emotion"] = np.array([0, 0, 1], dtype=np.float32)
        else:
            raise ValueError("Invalid emotion case.")

        stats_dict[relative_path]["arousal"] = np.array([0, 0, 0], dtype=np.float32)
        stats_dict[relative_path]["valence"] = np.array([0, 0, 0], dtype=np.float32)
        stats_dict[relative_path]["dominance"] = np.array([0, 0, 0], dtype=np.float32)

    return stats_dict


file_stats = read_stats(root_dir + "/stats.txt")


def get_portion_to_id(folder):
    p_t_i = dict()
    p_t_i["train"] = list()
    p_t_i["train_aug"] = list()
    p_t_i["valid"] = list()
    p_t_i["test"] = list()

    list_of_files = librosa.util.find_files(folder)
    for f in list_of_files:
        if "3-emo-augmented" in f:
            split_f = f.split("/")
            session_name = split_f[4]
            relative_file_path = "/".join(split_f[4:])
            # print(session_name)
            # print(relative_file_path)
            p_t_i["train_aug"].append(relative_file_path)
            # raise ValueError
            continue
        if not f[-4:] == ".wav":
            raise ValueError("IEMOCAP only has wavs.")
        split_f = f.split("/")
        session_name = split_f[4]
        relative_file_path = "/".join(split_f[4:])
        # print(relative_file_path)

        if relative_file_path not in file_stats.keys():
            continue

        # if split_f[-1][:-4] in MAX_TEST_SET:
        #     p_t_i["test"].append(relative_file_path)
        # else:
        #     p_t_i["train"].append(relative_file_path)

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

    # step_number = audio.size // (640 // 4)
    # step_number_remainder = audio.size % (640 // 4)
    # if step_number_remainder > 0:
    #     raise ValueError("This shouldn't be possible because it was corrected earlier.")
    #
    # audio = audio
    #
    # audio_frames = list()
    # for t in range(step_number-3):
    #     start_index = t*(640 // 4)
    #     end_index = t*(640 // 4) + 640
    #     audio_frames.append(audio[start_index:end_index])
    #
    # audio_frames = np.concatenate(audio_frames, axis=0)
    # print(audio_frames.shape)

    audio_frames = audio

    # step_number = audio.size // (640 // 4)
    # step_number_remainder = audio.size % (640 // 4)
    # # if step_number < 4:
    # #     raise ValueError("6")
    #
    # audio = audio[::-1]
    #
    # audio_frames = collections.deque()
    # for t in range(step_number-3):
    #     audio_frames.appendleft(audio[t*640:(t+1)*640][::-1])
    #
    # if step_number_remainder > 0:
    #     audio_frames.appendleft(np.concatenate([audio[-step_number_remainder:],
    #                                             np.zeros((640-step_number_remainder, ), dtype=np.float32)], axis=0)[::-1])
    #                                             # audio[-1] * np.ones((640-step_number_remainder, ), dtype=np.float32)], axis=0)[::-1])
    #     if audio_frames[-1].size != 640:
    #         print(audio_frames[-1].size)
    #         raise ValueError("uh huh")
    #
    # audio_frames = np.concatenate(audio_frames, axis=0)
    # print(audio_frames.shape)

    # print(np.sum(np.abs(audio[::-1][-640:] - audio_2[-640:])))
    # if step_number_remainder == 0:
    #     print(np.sum(np.abs(audio[::-1][:640] - audio_2[:640])))
    # else:
    #     print(np.sum(np.abs(audio[::-1][:step_number_remainder] - audio_2[640-step_number_remainder:(640)])))

    # emotion = [file_stats[file_path]["emotion"],] * len(audio_frames)
    # arousal = [file_stats[file_path]["arousal"],] * len(audio_frames)
    # valence = [file_stats[file_path]["valence"],] * len(audio_frames)
    # dominance = [file_stats[file_path]["dominance"],] * len(audio_frames)

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

    for relative_file_path in print_progress(portion_to_id[split]):
        audio = librosa.core.load(root_dir + "/" + relative_file_path, sr=48000, mono=False)

        audio = librosa.resample(audio[0], orig_sr=48000, target_sr=16000)

        if audio.size > max_seq_len:
            max_seq_len = audio.size

        sum_value += np.sum(audio)
        number_of_values += audio.size
    mean_value = sum_value / number_of_values

    sum_squares_value = 0.0
    for relative_file_path in print_progress(portion_to_id[split]):
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

        for relative_file_path in print_progress(portion_to_id[portion]):
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
    print(len(aa["train_aug"]))
    print(len(aa["valid"]))
    print(len(aa["test"]))

    main(Path("/data/Data/data_folder/preprocessed_data/IEMOCAP/speaker_independent/all"))
    # main(Path("/data/Data/data_folder/preprocessed_data/IEMOCAP/speaker_independent/only_improv"))
