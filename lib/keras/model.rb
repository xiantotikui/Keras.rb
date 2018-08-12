module Model
  extend self
  pyfrom 'keras.models', import: 'Sequential'

  def sequential(model = nil, **args)
    Sequential.new(model, **args)
  end
end
