module Keras::Recurrent
  extend self
  pyfrom 'keras.layers', import: 'RNN'
  pyfrom 'keras.layers', import: 'SimpleRNN'
  pyfrom 'keras.layers', import: 'GRU'
  pyfrom 'keras.layers', import: 'LSTM'
  pyfrom 'keras.layers', import: 'ConvLSTM2D'
  pyfrom 'keras.layers', import: 'SimpleRNNCell'
  pyfrom 'keras.layers', import: 'GRUCell'
  pyfrom 'keras.layers', import: 'LSTMCell'
  pyfrom 'keras.layers', import: 'CuDNNGRU'
  pyfrom 'keras.layers', import: 'CuDNNLSTM'

  def rnn(cell, **args)
    RNN.new(cell, **args)
  end

  def simple_rnn(units, **args)
    SimpleRNN.new(units, **args)
  end

  def gru(units, **args)
    GRU.new(units, **args)
  end

  def lstm(units, **args)
    LSTM.new(units, **args)
  end

  def conv_lstm2D(filters, kernel_size, **args)
    ConvLSTM2D.new(filters, kernel_size, **args)
  end

  def simple_rnn_cell(units, **args)
    SimpleRNNCell.new(units, **args)
  end

  def gru_cell(units, **args)
    GRUCell.new(units, **args)
  end

  def lstm_cell(units, **args)
    LSTMCell.new(units, **args)
  end

  def cuddn_gru(units, **args)
    CuDNNGRU.new(units, **args)
  end

  def cuddn_lstm(units, **args)
    CuDNNLSTM.new(units, **args)
  end
end
