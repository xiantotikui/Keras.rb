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

  def adagrad(lr: 0.01, **args)
    pyfrom 'keras.optimizers', import: 'Adagrad'
    Adagrad.new(lr: lr, **args)
  end

  def adadelta(lr: 1.0, **args)
    pyfrom 'keras.optimizers', import: 'Adadelta'
    Adadelta.new(lr: lr, **args)
  end

  def adam(lr: 0.001, **args)
    pyfrom 'keras.optimizers', import: 'Adam'
    Adam.new(lr: lr, **args)
  end

  def adamax(lr: 0.002, **args)
    pyfrom 'keras.optimizers', import: 'Adamax'
    Adamax.new(lr: lr, **args)
  end

  def nadam(lr: 0.002, **args)
    pyfrom 'keras.optimizers', import: 'Nadam'
    Nadam.new(lr: lr, **args)
  end

  def tfoptimizre(optimizer, **args)
    pyfrom 'keras.optimizers', import: 'TFOptimizer'
    TFOptimizer.new(optimizer, **args)
  end
end
