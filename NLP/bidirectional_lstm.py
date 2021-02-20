import tensorflow as tf

tf_fc = tf.contrib.feature_column


# Text classification model
class ClassificationModel(object):
    # Model initialization
    def __init__(self, vocab_size, max_length, num_lstm_units):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Make LSTM cell with dropout
    def make_lstm_cell(self, dropout_keep_prob):
        cell = tf.nn.rnn_cell.LSTMCell(self.num_lstm_units)
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)

    # Use feature columns to create input embeddings
    def get_input_embeddings(self, input_sequences):
        inputs_column = tf_fc.sequence_categorical_column_with_identity(
            'inputs',
            self.vocab_size)
        embedding_column = tf.feature_column.embedding_column(
            inputs_column,
            int(self.vocab_size ** 0.25))
        inputs_dict = {'inputs': input_sequences}
        input_embeddings, sequence_lengths = tf_fc.sequence_input_layer(
            inputs_dict,
            [embedding_column])
        return input_embeddings, sequence_lengths

    # Create and run a BiLSTM on the input sequences
    def run_bilstm(self, input_sequences, is_training):
        input_embeddings, sequence_lengths = self.get_input_embeddings(input_sequences)
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_fw = self.make_lstm_cell(dropout_keep_prob)
        cell_bw = self.make_lstm_cell(dropout_keep_prob)
        lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            input_embeddings,
            sequence_length=sequence_lengths,
            dtype=tf.float32)
        return lstm_outputs, sequence_lengths

    def get_gather_indices(self, batch_size, sequence_lengths):
        row_indices = tf.range(batch_size)
        final_indexes = tf.cast(sequence_lengths - 1, tf.int32)
        return tf.transpose([row_indices, final_indexes])

    # Calculate logits based on the outputs of the BiLSTM
    def calculate_logits(self, lstm_outputs, batch_size, sequence_lengths):
        lstm_outputs_fw, lstm_outputs_bw = lstm_outputs
        combined_outputs = tf.concat([lstm_outputs_fw, lstm_outputs_bw], -1)
        gather_indices = self.get_gather_indices(batch_size, sequence_lengths)
        final_outputs = tf.gather_nd(combined_outputs, gather_indices)
        logits = tf.layers.dense(final_outputs, 1)
        return logits

    # Calculate the loss for the BiLSTM
    def calculate_loss(self, lstm_outputs, batch_size, sequence_lengths, labels):
        logits = self.calculate_logits(lstm_outputs, batch_size, sequence_lengths)
        float_labels = tf.cast(labels, tf.float32)
        batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=float_labels, logits=logits)
        overall_loss = tf.reduce_sum(batch_loss)
        return overall_loss

    # Convert logits to predictions
    def logits_to_predictions(self, logits):
        probs = tf.nn.sigmoid(logits)
        preds = tf.round(probs)
        return preds
