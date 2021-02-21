import tensorflow as tf
from IndustryCaseStudy.data_processing import create_feature_columns, create_tensorflow_dataset
from IndustryCaseStudy.data_analysis import clean_data


class SalesModel(object):
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.final_dataset = clean_data()
        self.feature_columns = create_feature_columns(self.final_dataset)

    def run_regression_predict(self, ckpt_dir, data_file):
        regression_model = self.create_regression_model(ckpt_dir)
        input_fn = lambda: create_tensorflow_dataset(data_file, 1, training=False, has_labels=False)
        predictions = regression_model.predict(input_fn)
        pred_list = []
        for pred_dict in predictions:
            pred_list.append(pred_dict['predictions'][0])
        return pred_list

    def run_regression_eval(self, ckpt_dir):
        regression_model = self.create_regression_model(ckpt_dir)
        input_fn = lambda: create_tensorflow_dataset('eval.tfrecords', 50, training=False)
        return regression_model.evaluate(input_fn)

    def run_regression_training(self, ckpt_dir, batch_size, num_training_steps=None):
        regression_model = self.create_regression_model(ckpt_dir)
        input_fn = lambda: create_tensorflow_dataset('train.tfrecords', batch_size)
        regression_model.train(input_fn, steps=num_training_steps)

    def create_regression_model(self, ckpt_dir):
        config = tf.estimator.RunConfig(log_step_count_steps=5000)
        regression_model = tf.estimator.Estimator(
            self.regression_fn,
            config=config,
            model_dir=ckpt_dir)
        return regression_model

    def model_layers(self, inputs):
        layer = inputs
        for num_nodes in self.hidden_layers:
            layer = tf.layers.dense(layer, num_nodes, activation=tf.nn.relu)
        batch_predictions = tf.layers.dense(layer, 1)
        return batch_predictions

    def regression_fn(self, features, labels, mode, params):
        inputs = tf.feature_column.input_layer(features, self.feature_columns)
        batch_predictions = self.model_layers(inputs)
        predictions = tf.squeeze(batch_predictions)
        if labels is not None:
            loss = tf.losses.absolute_difference(labels, predictions)
        else:
            return

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            adam = tf.train.AdamOptimizer()
            train_op = adam.minimize(
                loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)

        if mode == tf.estimator.ModeKeys.PREDICT:
            prediction_info = {'predictions': batch_predictions}
            return tf.estimator.EstimatorSpec(mode, predictions=prediction_info)
