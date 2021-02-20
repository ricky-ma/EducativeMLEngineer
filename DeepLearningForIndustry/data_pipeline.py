import tensorflow as tf
import numpy as np


def dict_to_example(data_dict, config):
    feature_dict = {}
    for feature_name, value in data_dict.items():
        feature_config = config[feature_name]
        shape = feature_config['shape']
        if shape == () or shape == []:
            value = [value]
        value_type = feature_config['type']
        if value_type == 'int':
            feature_dict[feature_name] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=value))
        elif value_type == 'float':
            feature_dict[feature_name] = tf.train.Feature(
                float_list=tf.train.FloatList(value=value))
        elif value_type == 'string' or value_type == 'bytes':
            feature_dict[feature_name] = make_bytes_feature(
              value, value_type)
    features = tf.train.Features(feature=feature_dict)
    return tf.train.Example(features=features)


def make_bytes_feature(value, value_type):
    if value_type == 'string':
        value = [s.encode() for s in value]
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value))


def create_example_spec(config):
    example_spec = {}
    for feature_name, feature_config in config.items():
        if feature_config['type'] == 'int':
            tf_type = tf.int64
        elif feature_config['type'] == 'float':
            tf_type = tf.float32
        else:
            tf_type = tf.string
        shape = feature_config['shape']
        if shape is None:
            feature = tf.VarLenFeature(tf_type)
        else:
            default_value = feature_config.get('default_value', None)
            feature = tf.FixedLenFeature(shape, tf_type, default_value)
        example_spec[feature_name] = feature
    return example_spec


def parse_example(example_bytes, example_spec, output_features=None):
    parsed_features = tf.parse_single_example(example_bytes, example_spec)
    if output_features is not None:
        parsed_features = {k: parsed_features[k] for k in output_features}
    return parsed_features


def dataset_from_examples(filenames, config, output_features=None):
    example_spec = create_example_spec(config)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(
        lambda example: parse_example(example, example_spec, output_features))
    return dataset


def configure():
    data = np.random.uniform(-100, 100, (1000, 5))
    original = tf.data.Dataset.from_tensor_slices(data)

    shuffled1 = original.shuffle(100)
    print(shuffled1)
    shuffled2 = original.shuffle(len(data))
    print(shuffled2)

    repeat1 = original.repeat(1)
    print(repeat1)
    repeat2 = original.repeat(100)
    print(repeat2)
    repeat3 = original.repeat()
    print(repeat3)

    batch1 = original.batch(1)
    print(batch1)
    batch2 = original.batch(100)
    print(batch2)


def create_feature_columns(config, example_spec, output_features=None):
    if output_features is None:
        output_features = config.keys()
    feature_columns = []
    for feature_name in output_features:
        dtype = example_spec[feature_name].dtype
        feature_config = config[feature_name]
        # HELPER FUNCTIONS USED
        if 'vocab_list' in feature_config:
            feature_col = create_list_column(feature_name, feature_config, dtype)
        elif 'vocab_file' in feature_config:
            feature_col = create_file_column(feature_name, feature_config, dtype)
        else:
            feature_col = create_numeric_column(feature_name, feature_config, dtype)
        feature_columns.append(feature_col)
    return feature_columns


def create_list_column(feature_name, feature_config, dtype):
    vocab_feature_col = tf.feature_column.categorical_column_with_vocabulary_list(
        feature_name, feature_config['vocab_list'], dtype=dtype)
    feature_col = tf.feature_column.indicator_column(vocab_feature_col)
    return feature_col


def create_file_column(feature_name, feature_config, dtype):
    vocab_feature_col = tf.feature_column.categorical_column_with_vocabulary_file(
        feature_name, feature_config['vocab_file'], dtype=dtype)
    feature_col = tf.feature_column.indicator_column(vocab_feature_col)
    return feature_col


def create_numeric_column(feature_name, feature_config, dtype):
    feature_col = tf.feature_column.numeric_column(
        feature_name, shape=feature_config['shape'], dtype=dtype)
    return feature_col
