from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path

import tensorflow as tf


def get_split(dataset_dir,
              is_training,
              split_name_list,
              batch_size,
              seq_length,
              buffer_size):
    paths = list()

    for split_name in split_name_list:
        root_path = Path(dataset_dir) / split_name
        add_paths = [str(x) for x in root_path.glob('*.tfrecords')]
        paths.extend(add_paths)

    dataset = tf.data.TFRecordDataset(paths)
    dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                            features={
                                                                'subject_id': tf.FixedLenFeature([], tf.int64),
                                                                'emotion': tf.FixedLenFeature([], tf.string),
                                                                # 'emotion_shape': tf.FixedLenFeature([], tf.string),
                                                                'arousal': tf.FixedLenFeature([], tf.string),
                                                                # 'arousal_shape': tf.FixedLenFeature([], tf.string),
                                                                'valence': tf.FixedLenFeature([], tf.string),
                                                                # 'valence_shape': tf.FixedLenFeature([], tf.string),
                                                                'dominance': tf.FixedLenFeature([], tf.string),
                                                                # 'dominance_shape': tf.FixedLenFeature([], tf.string),
                                                                'raw_audio': tf.FixedLenFeature([], tf.string),
                                                                # 'raw_audio_shape': tf.FixedLenFeature([], tf.string),
                                                                'pre_pad_length': tf.FixedLenFeature([], tf.int64)
                                                            }
                                                            ))

    dataset = dataset.map(lambda x: {
                                     'subject_id': tf.cast(tf.reshape(x['subject_id'], (1, )), tf.int32),
                                     'emotion': tf.reshape(tf.decode_raw(x['emotion'], tf.float32), (3, )),
                                     # 'emotion_shape': tf.reshape(tf.decode_raw(x['emotion_shape'], tf.float32), (1, )),
                                     'arousal': tf.reshape(tf.decode_raw(x['arousal'], tf.float32), (3, )),
                                     # 'arousal_shape': tf.reshape(tf.decode_raw(x['arousal_shape'], tf.float32), (None, 2)),
                                     'valence': tf.reshape(tf.decode_raw(x['valence'], tf.float32), (3, )),
                                     # 'valence_shape': tf.reshape(tf.decode_raw(x['valence_shape'], tf.float32), (None, 2)),
                                     'dominance': tf.reshape(tf.decode_raw(x['dominance'], tf.float32), (3, )),
                                     # 'dominance_shape': tf.reshape(tf.decode_raw(x['dominance_shape'], tf.float32), (None, 2)),
                                     'raw_audio': tf.reshape(tf.decode_raw(x['raw_audio'], tf.float32), (510720, )),
                                     # 'raw_audio_shape': tf.reshape(tf.decode_raw(x['raw_audio_shape'], tf.float32), (None, 2))
                                     'pre_pad_length': tf.cast(tf.reshape(x['subject_id'], (1, )), tf.int32),
                                                                    })

    # dataset = dataset.repeat()
    # dataset = dataset.batch(seq_length)
    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)

    return dataset
