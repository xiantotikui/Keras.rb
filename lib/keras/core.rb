module Core
  pyimport :keras
  extend self

  def dense(val0, val1 = nil, val2 = nil, val3 = true, val4 = 'glorot_uniform', val5 = 'zeros', val6 = nil, val7 = nil, val8 = nil, val9 = nil, val10 = nil)
    # First dense must have two arguments
    ret = -> v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 { keras.layers.Dense.new(v0, input_dim: v1, activation: v2, use_bias: v3, kernel_initializer: v4, bias_initializer: v5, kernel_regularizer: v6, bias_regularizer: v7, activity_regularizer: v8, kernel_constraint: v9, bias_constraint: v10) }
    ret.(val0, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10)
  end

  def activation(val0)
    ret = -> v0 { keras.layers.Activation.new(v0) }
    ret.(val0)
  end

  def dropout(val0, val1 = nil, val2 = nil)
    ret = -> v0, v1, v2 { keras.layers.Dropout.new(v0, noise_shape: v1, seed: v2) }
    ret.(val0, val1, val2)
  end

  def flatten(val0 = nil)
    ret = -> v0 { keras.layers.Flatten.new(data_format: v0) }
    ret.(val0)
  end

  def reshape(var0)
    ret = -> v0 { keras.layers.Reshape.new(v0) }
    ret.(var0)
  end

  def permute(var0)
    ret = -> v0 { keras.layers.Permute.new(v0) }
    ret.(var0)
  end

  def repeat_vector(var0)
    ret = -> v0 { keras.layers.RepeatVector.new(v0) }
    ret.(var0)
  end

  def activity_regularization(var0 = 0.0, var1 = 0.0)
    ret = -> v0, v1 { keras.layers.ActivityRegularization.new(v0, v1) }
    ret.(var0, var1)
  end

  def masking(val0 = 0.0)
    ret = -> v0 { keras.layers.Masking.new(v0) }
    ret.(val0)
  end

  def spatial_dropout1D(val0)
    ret = -> v0 { keras.layers.SpatialDropout1D.new(v0) }
    ret.(val0)
  end

  def spatial_dropout2D(val0)
    ret = -> v0 { keras.layers.SpatialDropout2D.new(v0) }
    ret.(val0)
  end

  def spatial_dropout3D(val0)
    ret = -> v0 { keras.layers.SpatialDropout3D.new(v0) }
    ret.(val0)
  end
end
