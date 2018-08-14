module Keras::AdvancedActivations
  extend self
  pyfrom 'keras.layers', import: 'LeakyReLU'
  pyfrom 'keras.layers', import: 'PReLU'
  pyfrom 'keras.layers', import: 'ELU'
  pyfrom 'keras.layers', import: 'ThresholdedReLU'
  pyfrom 'keras.layers', import: 'Softmax'

  def leakyrelu(alpha: 0.3, **args)
    LeakyReLU.new(alpha: alpha, **args)
  end

  def prelu(alpha_initializer: 'zeros', **args)
    PReLU.new(alpha_initializer: alpha_initializer, **args)
  end

  def elu(alpha: 1.0, **args)
    ELU.new(alpha: alpha, **args)
  end

  def thresholdedrelu(theta: 1.0, **args)
    ThresholdedReLU.new(theta: theta, **args)
  end

  def softmax(axis: -1.0, **args)
    Softmax.new(axis: axis, **args)
  end
end
