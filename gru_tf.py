import tensorflow as tf

class GRU_Model(tf.keras.Model):

  def __init__(self, vocab_size, embedding_dim, units):

    super(GRU_Model, self).__init__()

    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    if tf.test.is_gpu_available():
      self.gru = tf.keras.layers.CuDNNGRU(self.units,
                                          return_sequences=True,
                                          recurrent_initializer='glorot_uniform',
                                          stateful=True)
    else:
      self.gru = tf.keras.layers.GRU(self.units,
                                     return_sequences=True,
                                     recurrent_activation='sigmoid',
                                     recurrent_initializer='glorot_uniform',
                                     stateful=True)

    self.fc = tf.keras.layers.Dense(vocab_size)

  def call(self, x):
    embedding = self.embedding(x)

    # output at every time step
    # output shape == (batch_size, seq_length, hidden_size)
    output = self.gru(embedding)

    # The dense layer will output predictions for every time_steps(seq_length)
    # output shape after the dense layer == (seq_length * batch_size, vocab_size)
    prediction = self.fc(output)

    # states will be used to pass at every step to the model while training
    return prediction

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# Using sparse_softmax_cross_entropy
def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)
