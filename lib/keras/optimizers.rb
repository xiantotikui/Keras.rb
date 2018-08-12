module Optimizers
  extend self

  def sgd(learning_rate: 0.01, **args)
    pyfrom 'keras.optimizers', import: 'SGD'
    SGD.new(lr: learning_rate, **args)
  end

  def rmsprop(learning_rate: 0.001, **args)
    pyfrom 'keras.optimizers', import: 'RMSprop'
    RMSprop.new(lr: learning_rate, **args)
  end
end
