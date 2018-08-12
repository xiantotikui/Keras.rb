module Core
  extend self
  pyfrom 'keras.layers', import: 'Dense'
  pyfrom 'keras.layers', import: 'Activation'
  pyfrom 'keras.layers', import: 'Dropout'
  pyfrom 'keras.layers', import: 'Flatten'
  pyfrom 'keras.layers', import: 'Reshape'
  pyfrom 'keras.layers', import: 'Permute'
  pyfrom 'keras.layers', import: 'RepeatVector'
  pyfrom 'keras.layers', import: 'ActivityRegularization'
  pyfrom 'keras.layers', import: 'Masking'
  pyfrom 'keras.layers', import: 'SpatialDropout1D'
  pyfrom 'keras.layers', import: 'SpatialDropout2D'
  pyfrom 'keras.layers', import: 'SpatialDropout3D'

  def dense(units, **args)
    Dense.new(units, **args)
  end

  def activation(activ, **args)
    Activation.new(activ, **args)
  end

  def dropout(rate, **args)
    Dropout.new(rate, **args)
  end

  def flatten(data_format: nil, **args)
    Flatten.new(data_format, **args)
  end

  def reshape(target_shape, **args)
    Reshape.new(target_shape, **args)
  end

  def permute(dims, **args)
    Permute.new(dims, **args)
  end

  def repeat_vector(n, **args)
    RepeatVector.new(n, **args)
  end

  def activity_regularization(l1 = 0.0, l2 = 0.0, **args)
    ActivityRegularization.new(l1, l2, **args)
  end

  def masking(mask_value = 0.0, **args)
    Masking.new(mask_value, **args)
  end

  def spatial_dropout1D(rate, **args)
    SpatialDropout1D.new(rate, **args)
  end

  def spatial_dropout2D(rate, **args)
    SpatialDropout2D.new(rate, **args)
  end

  def spatial_dropout3D(rate, **args)
    SpatialDropout3D.new(rate, **args)
  end
end
