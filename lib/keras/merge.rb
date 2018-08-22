module Keras::Merge
  extend self
  pyfrom 'keras.layers', import: 'add'
  pyfrom 'keras.layers', import: 'subtract'
  pyfrom 'keras.layers', import: 'multiply'
  pyfrom 'keras.layers', import: 'average'
  pyfrom 'keras.layers', import: 'maximum'
  pyfrom 'keras.layers', import: 'minimum'
  pyfrom 'keras.layers', import: 'concatenate'
  pyfrom 'keras.layers', import: 'dot'

  def add(inputs, **args)
    add(inputs, **args)
  end

  def subtract(inputs, **args)
    subtract(inputs, **args)
  end

  def multiply(inputs, **args)
    multiply(inputs, **args)
  end

  def average(inputs, **args)
    average(inputs, **args)
  end

  def maximum(inputs, **args)
    maximum(inputs, **args)
  end

  def minimum(inputs, **args)
    minimum(inputs, **args)
  end

  def concatenate(inputs, **args)
    concatenate(inputs, **args)
  end

  def dot(inputs, **args)
    dot(inputs, **args)
  end
end
