import tensorflow as tf

tf_fc = tf.contrib.feature_column
tf_s2s = tf.contrib.seq2seq


# Seq2seq model
class Seq2SeqModel(object):
    def __init__(self, vocab_size, num_lstm_layers, num_lstm_units):
        self.vocab_size = vocab_size
        # Extended vocabulary includes start, stop token
        self.extended_vocab_size = vocab_size + 2
        self.num_lstm_layers = num_lstm_layers
        self.num_lstm_units = num_lstm_units
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size)

    # Create a sequence training tuple from input/output sequences
    def make_training_tuple(self, input_sequence, output_sequence):
        truncate_front = output_sequence[1:]
        truncate_back = output_sequence[:-1]
        sos_token = [self.vocab_size]
        eos_token = [self.vocab_size + 1]
        input_sequence = sos_token + input_sequence + eos_token
        ground_truth = sos_token + truncate_back
        final_sequence = truncate_front + eos_token
        return input_sequence, ground_truth, final_sequence

    def make_lstm_cell(self, dropout_keep_prob, num_units):
        cell = tf.nn.rnn_cell.LSTMCell(num_units)
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)

    # Create multi-layer LSTM
    def stacked_lstm_cells(self, is_training, num_units):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.make_lstm_cell(dropout_keep_prob, num_units) for i in range(self.num_lstm_layers)]
        cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
        return cell

    # Get embeddings for input/output sequences
    def get_embeddings(self, sequences, scope_name):
        with tf.variable_scope(scope_name):
            cat_column = tf_fc.sequence_categorical_column_with_identity(
                'sequences',
                self.extended_vocab_size)
            embedding_column = tf.feature_column.embedding_column(
                cat_column,
                int(self.extended_vocab_size ** 0.25))
            seq_dict = {'sequences': sequences}
            embeddings, sequence_lengths = tf_fc.sequence_input_layer(
                seq_dict,
                [embedding_column])
            return embeddings, tf.cast(sequence_lengths, tf.int32)

    # Get c and h vectors for bidirectional LSTM final states
    def get_bi_state_parts(self, state_fw, state_bw):
        bi_state_c = tf.concat([state_fw.c, state_bw.c], -1)
        bi_state_h = tf.concat([state_fw.h, state_bw.h], -1)
        return bi_state_c, bi_state_h

    # Create the encoder for the model
    def encoder(self, encoder_inputs, is_training):
        input_embeddings, input_seq_lens = self.get_embeddings(encoder_inputs, 'encoder_emb')
        cell_fw = self.stacked_lstm_cells(is_training, self.num_lstm_units)
        cell_bw = self.stacked_lstm_cells(is_training, self.num_lstm_units)
        enc_outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            input_embeddings,
            sequence_length=input_seq_lens,
            dtype=tf.float32)
        states_fw, states_bw = final_states
        combined_state = []
        for i in range(self.num_lstm_layers):
            bi_state_c, bi_state_h = self.get_bi_state_parts(
                states_fw[i], states_bw[i])
            bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(bi_state_c, bi_state_h)
            combined_state.append(bi_lstm_state)
        final_state = tuple(combined_state)
        return enc_outputs, input_seq_lens, final_state

    # Helper funtion to combine BiLSTM encoder outputs
    def combine_enc_outputs(self, enc_outputs):
        enc_outputs_fw, enc_outputs_bw = enc_outputs
        return tf.concat([enc_outputs_fw, enc_outputs_bw], -1)

    # Create the stacked LSTM cells for the decoder
    def create_decoder_cell(self, enc_outputs, input_seq_lens, is_training):
        num_decode_units = self.num_lstm_units * 2
        dec_cell = self.stacked_lstm_cells(is_training, num_decode_units)
        combined_enc_outputs = self.combine_enc_outputs(enc_outputs)
        attention_mechanism = tf_s2s.LuongAttention(
            num_decode_units, combined_enc_outputs,
            memory_sequence_length=input_seq_lens)
        dec_cell = tf_s2s.AttentionWrapper(
            dec_cell, attention_mechanism,
            attention_layer_size=num_decode_units)
        return dec_cell

    # Create the helper for decoding
    def create_decoder_helper(self, decoder_inputs, is_training, batch_size):
        if is_training:
            dec_embeddings, dec_seq_lens = self.get_embeddings(decoder_inputs, 'decoder_emb')
            helper = tf_s2s.TrainingHelper(
                dec_embeddings, dec_seq_lens)
        else:
            DEC_EMB_SCOPE = 'decoder_emb/sequence_input_layer/sequences_embedding'
            with tf.variable_scope(DEC_EMB_SCOPE):
                embedding_weights = tf.get_variable(
                    'embedding_weights',
                    shape=(self.extended_vocab_size, int(self.extended_vocab_size ** 0.25)))
            start_tokens = tf.tile([self.vocab_size], [batch_size])
            end_token = self.vocab_size + 1
            helper = tf_s2s.GreedyEmbeddingHelper(
                embedding_weights,
                start_tokens,
                end_token)
            dec_seq_lens = None
        return helper, dec_seq_lens

    def run_decoder(self, decoder, maximum_iterations, dec_seq_lens, is_training):
        dec_outputs, _, _ = tf_s2s.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
        if is_training:
            logits = dec_outputs.rnn_output
            return logits, dec_seq_lens
        return dec_outputs.sample_id

    # Create the decoder for the model
    def decoder(self, enc_outputs, input_seq_lens, final_state, batch_size,
                decoder_inputs=None, maximum_iterations=None):
        is_training = decoder_inputs is not None
        dec_cell = self.create_decoder_cell(enc_outputs, input_seq_lens, is_training)
        helper, dec_seq_lens = self.create_decoder_helper(decoder_inputs, is_training, batch_size)
        projection_layer = tf.layers.Dense(self.extended_vocab_size)
        zero_cell = dec_cell.zero_state(batch_size, tf.float32)
        initial_state = zero_cell.clone(cell_state=final_state)
        decoder = tf_s2s.BasicDecoder(
            dec_cell, helper, initial_state,
            output_layer=projection_layer)
        return self.run_decoder(decoder, maximum_iterations, dec_seq_lens, is_training)

    # Calculate the model loss
    def calculate_loss(self, logits, dec_seq_lens, decoder_outputs, batch_size):
        binary_sequences = tf.sequence_mask(dec_seq_lens, dtype=tf.float32)
        batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=decoder_outputs, logits=logits)
        unpadded_loss = batch_loss * binary_sequences
        per_seq_loss = tf.reduce_sum(unpadded_loss) / batch_size
        return per_seq_loss
