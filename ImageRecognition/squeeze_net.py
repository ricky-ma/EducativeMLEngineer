import tensorflow as tf


class SqueezeNetModel(object):
    # Model Initialization
    def __init__(self, original_dim, resize_dim, output_size):
        self.original_dim = original_dim
        self.resize_dim = resize_dim
        self.output_size = output_size

    # Convolution layer wrapper
    def custom_conv2d(self, inputs, filters, kernel_size, name):
        return tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            activation=tf.nn.relu,
            padding='same',
            name=name)

    # Max pooling layer wrapper
    def custom_max_pooling2d(self, inputs, name):
        return tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=[2, 2],
            strides=2,
            name=name)

    # SqueezeNet fire module
    def fire_module(self, inputs, squeeze_depth, expand_depth, name):
        with tf.variable_scope(name):
            squeezed_inputs = self.custom_conv2d(
                inputs,
                squeeze_depth,
                [1, 1],
                'squeeze')
            expand1x1 = self.custom_conv2d(
                squeezed_inputs,
                expand_depth,
                [1, 1],
                'expand1x1')
            expand3x3 = self.custom_conv2d(
                squeezed_inputs,
                expand_depth,
                [3, 3],
                'expand3x3')
            return tf.concat([expand1x1, expand3x3], axis=-1)

    # Utility function for multiple fire modules
    def multi_fire_module(self, layer, params_list):
        for params in params_list:
            layer = self.fire_module(
                layer,
                params[0],
                params[1],
                params[2])
        return layer

    # Model Layers
    # inputs: [batch_size, resize_dim, resize_dim, 3]
    def model_layers(self, inputs, is_training):
        conv1 = self.custom_conv2d(
            inputs,
            64,
            [3, 3],
            'conv1')
        pool1 = self.custom_max_pooling2d(
            conv1,
            'pool1')
        fire_params1 = [
            (32, 64, 'fire1'),
            (32, 64, 'fire2')
        ]
        multi_fire1 = self.multi_fire_module(
            pool1,
            fire_params1)
        pool2 = self.custom_max_pooling2d(
            multi_fire1,
            'pool2')
        fire_params2 = [
            (32, 128, 'fire3'),
            (32, 128, 'fire4')
        ]
        multi_fire2 = self.multi_fire_module(
            pool2,
            fire_params2)
        dropout1 = tf.layers.dropout(multi_fire2, rate=0.5,
                                     training=is_training)
        final_conv_layer = self.custom_conv2d(
            dropout1,
            self.output_size,
            [1, 1],
            'final_conv')
        avg_pool1 = tf.layers.average_pooling2d(
            final_conv_layer,
            [final_conv_layer.shape[1], final_conv_layer.shape[2]],
            1)
        logits = tf.layers.flatten(avg_pool1, name='logits')
        return logits

    # Set up and run model training
    def run_model_setup(self, inputs, labels, is_training):
        logits = self.model_layers(inputs, is_training)
        self.probs = tf.nn.softmax(logits, name='probs')
        self.predictions = tf.argmax(
            self.probs, axis=-1, name='predictions')
        is_correct = tf.equal(
            tf.cast(self.predictions, tf.int32),
            labels)
        is_correct_float = tf.cast(
            is_correct,
            tf.float32)
        self.accuracy = tf.reduce_mean(
            is_correct_float)
        # calculate cross entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)
        self.loss = tf.reduce_mean(
            cross_entropy)
        adam = tf.train.AdamOptimizer()
        self.train_op = adam.minimize(
            self.loss, global_step=self.global_step)
