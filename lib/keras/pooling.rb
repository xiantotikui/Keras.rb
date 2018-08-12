module Pooling
  extend self
  pyfrom 'keras.layers', import: 'MaxPooling1D'
  pyfrom 'keras.layers', import: 'MaxPooling2D'
  pyfrom 'keras.layers', import: 'MaxPooling3D'

  def max_pooling1D(ps, **args)
    MaxPooling1D.new(pool_size: ps, **args)
  end

  def max_pooling2D(ps, **args)
    MaxPooling2D.new(pool_size: ps, **args)
  end

  def max_pooling3D(ps, **args)
    MaxPooling3D.new(pool_size: ps, **args)
  end
end
