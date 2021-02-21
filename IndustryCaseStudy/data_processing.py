import tensorflow as tf


# Split the final pandas DataFrame into training and evaluation sets
def split_train_eval(final_dataset):
    final_dataset = final_dataset.sample(frac=1)
    eval_size = len(final_dataset) // 10
    eval_set = final_dataset.iloc[:eval_size]
    train_set = final_dataset.iloc[eval_size:]
    return train_set, eval_set


# Add the integer Feature objects to the feature dictionary
def add_int_features(dataset_row, feature_dict):
    int_vals = ['Store', 'Dept', 'IsHoliday', 'Size']
    for feature_name in int_vals:
        list_val = tf.train.Int64List(value=[dataset_row[feature_name]])
        feature_dict[feature_name] = tf.train.Feature(int64_list=list_val)


# Add the float Feature objects to the feature dictionary
def add_float_features(dataset_row, feature_dict, has_labels):
    float_vals = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    if has_labels:
        float_vals.append('Weekly_Sales')
    for feature_name in float_vals:
        list_val = tf.train.FloatList(value=[dataset_row[feature_name]])
        feature_dict[feature_name] = tf.train.Feature(float_list=list_val)


# Create an Example object from a pandas DataFrame row
def create_example(dataset_row, has_labels):
    feature_dict = {}
    add_int_features(dataset_row, feature_dict)
    add_float_features(dataset_row, feature_dict, has_labels)
    byte_type = dataset_row['Type'].encode()
    list_val = tf.train.BytesList(value=[byte_type])
    feature_dict['Type'] = tf.train.Feature(bytes_list=list_val)
    features_obj = tf.train.Features(feature=feature_dict)
    return tf.train.Example(features=features_obj)


# Write serialized Example objects to a TFRecords file
def write_tfrecords(dataset, has_labels, tfrecords_file):
    writer = tf.python_io.TFRecordWriter(tfrecords_file)
    for i in range(len(dataset)):
        example = create_example(dataset.iloc[i], has_labels)
        writer.write(example.SerializeToString())
    writer.close()


# Create the spec used when parsing the Example object
def create_example_spec(has_labels):
    example_spec = {}
    int_vals = ['Store', 'Dept', 'IsHoliday', 'Size']
    float_vals = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    if has_labels:
        float_vals.append('Weekly_Sales')
    for feature_name in int_vals:
        example_spec[feature_name] = tf.FixedLenFeature((), tf.int64)
    for feature_name in float_vals:
        example_spec[feature_name] = tf.FixedLenFeature((), tf.float32)
    example_spec['Type'] = tf.FixedLenFeature((), tf.string)
    return example_spec


# Helper function to convert serialized Example objects into features
def parse_features(ser_ex, example_spec, has_labels):
    parsed_features = tf.parse_single_example(ser_ex, example_spec)
    features = {k: parsed_features[k] for k in parsed_features if k != 'Weekly_Sales'}
    if not has_labels:
        return features
    label = parsed_features['Weekly_Sales']
    return features, label


# Load and configure dataset from TFRecords files
def load_dataset(train_file='train.tfrecords', eval_file='eval.tfrecords'):
    train_dataset = tf.data.TFRecordDataset(train_file)
    eval_dataset = tf.data.TFRecordDataset(eval_file)

    # Load example spec for dataset
    example_spec = create_example_spec(True)
    parse_fn = lambda ser_ex: parse_features(ser_ex, example_spec, True)
    train_dataset = train_dataset.map(parse_fn)
    eval_dataset = eval_dataset.map(parse_fn)

    # Configure dataset
    train_dataset = train_dataset.shuffle(421570)
    eval_dataset = eval_dataset.shuffle(421570)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(100)
    eval_dataset = eval_dataset.batch(20)

    return train_dataset, eval_dataset


def create_tensorflow_dataset(file, batch_size=50, shuffle=True, training=True, count=1000, has_labels=True):
    dataset = tf.data.TFRecordDataset(file)
    example_spec = create_example_spec(has_labels)
    parse_fn = lambda ser_ex: parse_features(ser_ex, example_spec, True)
    dataset = dataset.map(parse_fn)
    dataset = dataset.batch(batch_size)
    if shuffle: dataset = dataset.shuffle(421570)
    if training: dataset = dataset.repeat(count=count)
    return dataset


# Add numeric feature columns to a list of dataset feature columns
def add_numeric_columns(feature_columns):
    numeric_features = ['Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for feature_name in numeric_features:
        feature_col = tf.feature_column.numeric_column(feature_name, shape=())
        feature_columns.append(feature_col)
    return feature_columns


# Add the indicator feature columns to a list of feature columns
def add_indicator_columns(final_dataset, feature_columns):
    indicator_features = ['IsHoliday', 'Type']
    for feature_name in indicator_features:
        dtype = tf.string if feature_name == 'Type' else tf.int64
        vocab_list = list(final_dataset[feature_name].unique())
        vocab_col = tf.feature_column.categorical_column_with_vocabulary_list(
            feature_name, vocab_list, dtype=dtype)
        feature_col = tf.feature_column.indicator_column(vocab_col)
        feature_columns.append(feature_col)
    return feature_columns


# Add the embedding feature columns to a list of feature columns
def add_embedding_columns(final_dataset, feature_columns):
    embedding_features = ['Store', 'Dept']
    for feature_name in embedding_features:
        vocab_list = list(final_dataset[feature_name].unique())
        vocab_feature_col = tf.feature_column.categorical_column_with_vocabulary_list(
            feature_name, vocab_list, dtype=tf.int64)
        embedding_dim = int(len(vocab_list) ** 0.25)
        feature_col = tf.feature_column.embedding_column(vocab_feature_col, embedding_dim)
        feature_columns.append(feature_col)
    return feature_columns


def create_feature_columns(final_dataset):
    feature_columns = []
    feature_columns = add_numeric_columns(feature_columns)
    feature_columns = add_indicator_columns(final_dataset, feature_columns)
    feature_columns = add_embedding_columns(final_dataset, feature_columns)
    return feature_columns
