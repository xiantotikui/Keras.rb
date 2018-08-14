module Keras::Wrappers
  extend self
  pyfrom 'keras.layers', import: 'TimeDistributed'
  pyfrom 'keras.layers', import: 'Bidirectional'

  def time_distributed(layer, **args)
    TimeDistributed.new(layer, **args)
  end

  def bidirectional(layer, **args)
    Bidirectional.new(layer, **args)
  end
end
