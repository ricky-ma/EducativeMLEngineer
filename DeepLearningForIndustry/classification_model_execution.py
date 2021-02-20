import tensorflow as tf


class ClassificationModel(object):
    def __init__(self, output_size):
        self.output_size = output_size

    # Helper for make_predictions
    def load_inference_parts(self, sess, saved_model_dir):
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        inputs = sess.graph.get_tensor_by_name('inference_input:0')
        predictions_tensor = sess.graph.get_tensor_by_name('predictions:0')
        probs_tensor = sess.graph.get_tensor_by_name('probs:0')
        return inputs, predictions_tensor, probs_tensor

    # Make predictions with the inference model
    def make_predictions(self, saved_model_dir, input_data):
        with tf.Session(graph=tf.Graph()) as sess:
            inputs, predictions_tensor, probs_tensor = self.load_inference_parts(sess, saved_model_dir)
            predictions, probs = sess.run((predictions_tensor, probs_tensor), feed_dict={inputs: input_data})
        return predictions, probs

    # See the "Efficient Data Processing Techniques" section for details
    def dataset_from_numpy(self, input_data, batch_size, labels=None, is_training=True, num_epochs=None):
        dataset_input = input_data if labels is None else (input_data, labels)
        dataset = tf.data.Dataset.from_tensor_slices(dataset_input)
        if is_training:
            dataset = dataset.shuffle(len(input_data)).repeat(num_epochs)
        return dataset.batch(batch_size)

    # See the "Machine Learning for Software Engineers" course on Educative
    def run_model_setup(self, inputs, labels, hidden_layers, is_training, calculate_accuracy=True):
        layer = inputs
        for num_nodes in hidden_layers:
            layer = tf.layers.dense(layer, num_nodes,
                                    activation=tf.nn.relu)
        logits = tf.layers.dense(layer, self.output_size,
                                 name='logits')
        self.probs = tf.nn.softmax(logits, name='probs')
        self.predictions = tf.argmax(
            self.probs, axis=-1, name='predictions')
        if calculate_accuracy:
            class_labels = tf.argmax(labels, axis=-1)
            is_correct = tf.equal(
                self.predictions, class_labels)
            is_correct_float = tf.cast(
                is_correct,
                tf.float32)
            self.accuracy = tf.reduce_mean(
                is_correct_float)
        if labels is not None:
            labels_float = tf.cast(
                labels, tf.float32)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels_float,
                logits=logits)
            self.loss = tf.reduce_mean(
                cross_entropy)
        if is_training:
            adam = tf.train.AdamOptimizer()
            self.train_op = adam.minimize(
                self.loss, global_step=self.global_step)

    # Run training of the classification model
    def run_model_training(self, input_data, labels, hidden_layers, batch_size, num_epochs, ckpt_dir):
        self.global_step = tf.train.get_or_create_global_step()
        dataset = self.dataset_from_numpy(input_data, batch_size,
                                          labels=labels, num_epochs=num_epochs)
        iterator = dataset.make_one_shot_iterator()
        inputs, labels = iterator.get_next()
        self.run_model_setup(inputs, labels, hidden_layers, True)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.histogram('inputs', inputs)
        log_vals = {'loss': self.loss, 'step': self.global_step}
        logging_hook = tf.train.LoggingTensorHook(
            log_vals, every_n_iter=1000)
        nan_hook = tf.train.NanTensorHook(self.loss)
        hooks = [nan_hook, logging_hook]
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=ckpt_dir,
                hooks=hooks) as sess:
            while not sess.should_stop():
                sess.run(self.train_op)