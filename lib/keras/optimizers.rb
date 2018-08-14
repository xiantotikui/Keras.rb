module Keras::Optimizers
  extend self

  def sgd(lr: 0.01, **args)
    pyfrom 'keras.optimizers', import: 'SGD'
    SGD.new(lr: lr, **args)
  end

  def rmsprop(lr: 0.001, **args)
    pyfrom 'keras.optimizers', import: 'RMSprop'
    RMSprop.new(lr: lr, **args)
  end
end
