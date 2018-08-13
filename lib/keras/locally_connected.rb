module Keras::LocallyConnected
  extend self
  pyfrom 'keras.layers', import: 'LocallyConnected1D'
  pyfrom 'keras.layers', import: 'LocallyConnected2D'

  def locally_connected1D(filters, kernel_size, **args)
    LocallyConnected1D.new(filters, kernel_size, **args)
  end

  def locally_connected2D(filters, kernel_size, **args)
    LocallyConnected2D.new(filters, kernel_size, **args)
  end
end
