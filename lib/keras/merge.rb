module Keras::Merge
  extend self
  pyfrom 'keras.layers', import: 'Add'
  pyfrom 'keras.layers', import: 'Subtract'
  pyfrom 'keras.layers', import: 'Multiply'
  pyfrom 'keras.layers', import: 'Average'
  pyfrom 'keras.layers', import: 'Maximum'
  pyfrom 'keras.layers', import: 'Minimum'
  pyfrom 'keras.layers', import: 'Concatenate'
  pyfrom 'keras.layers', import: 'Dot'

  def add(inputs, **args)
    Add.new(inputs, **args)
  end

  def subtract(inputs, **args)
    Subtract.new(inputs, **args)
  end

  def multiply(inputs, **args)
    Multiply.new(inputs, **args)
  end

  def average(inputs, **args)
    Average.new(inputs, **args)
  end

  def maximum(inputs, **args)
    Maximum.new(inputs, **args)
  end

  def minimum(inputs, **args)
    Minimum.new(inputs, **args)
  end

  def concatenate(inputs, **args)
    Concatenate.new(inputs, **args)
  end

  def dot(inputs, **args)
    Dot.new(inputs, **args)
  end
end
