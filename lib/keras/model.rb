module Keras::Model
  extend self
  pyfrom 'keras.models', import: 'Sequential'
  pyfrom 'keras.models', import: 'Model'
  pyfrom 'keras.models', import: 'load_model'

  def sequential(model = nil, **args)
    Sequential.new(model, **args)
  end

  def model(input: nil, output: nil, **args)
    Model.new(input: input, output: output, **args)
  end

  def load(filepath)
    load_model(filepath)
  end
end
