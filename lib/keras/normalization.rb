module Keras::Normalization
  extend self
  pyfrom 'keras.layers', import: 'BatchNormalization'

  def batch_normalization(axis: -1, **args)
    BatchNormalization.new(axis: axis, **args)
  end
end
