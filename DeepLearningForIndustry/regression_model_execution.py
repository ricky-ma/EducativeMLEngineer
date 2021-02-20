import tensorflow as tf


class RegressionModel(object):
    def __init__(self, output_size):
        self.output_size = output_size

    # Helper for regressor_fn
    def eval_regressor(self, mode, labels):
        mse_metric = tf.metrics.mean_squared_error(labels, self.predictions)
        eval_metric = {'mse': mse_metric}
        estimator_spec = tf.estimator.EstimatorSpec(
            mode, loss=self.loss, eval_metric_ops=eval_metric)
        return estimator_spec

    # Helper from previous chapter
    def set_predictions_and_loss(self, logits, labels):
        self.predictions = tf.squeeze(logits)
        if labels is not None:
            self.loss = tf.nn.l2_loss(labels - self.predictions)

    # The function for the regression model
    def regressor_fn(self, features, labels, mode, params):
        inputs = tf.feature_column.input_layer(features, params['feature_columns'])
        layer = inputs
        for num_nodes in params['hidden_layers']:
            layer = tf.layers.dense(layer, num_nodes,
                                    activation=tf.nn.relu)
        logits = tf.layers.dense(layer, self.output_size,
                                 name='logits')
        self.set_predictions_and_loss(logits, labels)
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.global_step = tf.train.get_or_create_global_step()
            adam = tf.train.AdamOptimizer()
            self.train_op = adam.minimize(
                self.loss, global_step=self.global_step)
            return tf.estimator.EstimatorSpec(mode,
                                              loss=self.loss, train_op=self.train_op)
        if mode == tf.estimator.ModeKeys.EVAL:
            return self.eval_regressor(mode, labels)
        if mode == tf.estimator.ModeKeys.PREDICT:
            pass

    def dataset_from_examples(self, filenames, example_spec, batch_size,
                              buffer_size=None, use_labels=True, num_epochs=None):
        dataset = tf.data.TFRecordDataset(filenames)

        def _parse_fn(example_bytes):
            parsed_features = tf.parse_single_example(example_bytes, example_spec)
            label = parsed_features['label']
            output_features = [k for k in parsed_features.keys() if k != 'label']
            if use_labels:
                return {k: parsed_features[k] for k in output_features}, label
            return {k: parsed_features[k] for k in output_features}

        dataset = dataset.map(_parse_fn)
        if buffer_size is not None:
            dataset = dataset.shuffle(buffer_size)
        return dataset.repeat(num_epochs).batch(batch_size)

    def run_regressor_training(self, ckpt_dir, hidden_layers, feature_columns, filenames,
                               example_spec, batch_size, num_examples, num_training_steps=None):
        params = {
            'feature_columns': feature_columns,
            'hidden_layers': hidden_layers
        }
        regressor = tf.estimator.Estimator(
            self.regressor_fn,
            model_dir=ckpt_dir,
            params=params)
        input_fn = lambda: self.dataset_from_examples(
            filenames, example_spec, batch_size, buffer_size=num_examples)
        train_dict = regressor.train(
            input_fn,
            steps=num_training_steps)
        eval_dict = regressor.evaluate(
            input_fn,  # lambda function
            steps=2)
        preds = regressor.predict(
            input_fn,
            predict_keys=['prediction'])
        return train_dict, eval_dict, preds
