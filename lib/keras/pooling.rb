module Keras::Pooling
  extend self
  pyfrom 'keras.layers', import: 'MaxPooling1D'
  pyfrom 'keras.layers', import: 'MaxPooling2D'
  pyfrom 'keras.layers', import: 'MaxPooling3D'
  pyfrom 'keras.layers', import: 'AveragePooling1D'
  pyfrom 'keras.layers', import: 'AveragePooling2D'
  pyfrom 'keras.layers', import: 'AveragePooling3D'
  pyfrom 'keras.layers', import: 'GlobalMaxPooling1D'
  pyfrom 'keras.layers', import: 'GlobalAveragePooling1D'
  pyfrom 'keras.layers', import: 'GlobalMaxPooling2D'
  pyfrom 'keras.layers', import: 'GlobalAveragePooling2D'
  pyfrom 'keras.layers', import: 'GlobalMaxPooling3D'
  pyfrom 'keras.layers', import: 'GlobalAveragePooling3D'

  def max_pooling1D(pool_size: 2, **args)
    MaxPooling1D.new(pool_size: pool_size, **args)
  end

  def max_pooling2D(pool_size: [2, 2], **args)
    MaxPooling2D.new(pool_size: pool_size, **args)
  end

  def max_pooling3D(pool_size: [2, 2, 2], **args)
    MaxPooling3D.new(pool_size: pool_size, **args)
  end

  def average_pooling1D(pool_size: 2, **args)
    AveragePooling1D.new(pool_size: pool_size, **args)
  end

  def average_pooling2D(pool_size: [2, 2], **args)
    AveragePooling2D.new(pool_size: pool_size, **args)
  end

  def average_pooling3D(pool_size: [2, 2, 2], **args)
    AveragePooling3D.new(pool_size: pool_size, **args)
  end

  def global_max_pooling1D
    GlobalMaxPooling1D.new
  end

  def global_average_pooling1D
    GlobalAveragePooling1D.new
  end

  def global_max_pooling2D(data_format: nil, **args)
    GlobalMaxPooling2D.new(data_format: data_format, **args)
  end

  def global_average_pooling2D(data_format: nil, **args)
    GlobalAveragePooling2D.new(data_format: data_format, **args)
  end

  def global_max_pooling3D(data_format: nil, **args)
    GlobalMaxPooling3D.new(data_format: data_format, **args)
  end

  def global_average_pooling3D(data_format: nil, **args)
    GlobalAveragePooling3D.new(data_format: data_format, **args)
  end
end
