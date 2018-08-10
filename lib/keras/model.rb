module Model
  pyimport :keras
  extend self

  def sequential
    ret = -> { keras.Sequential.new }
    ret.()
  end

  def model(val0, val1)
    ret = -> v0, v1 { keras.Model.new(inputs: v0, outputs: v1) }
    ret.(val0, val1)
  end

  def compile(val0, val1, val2 = nil, val3 = nil, val4 = nil, val5 = nil, val6 =nil)
    ret = -> v0, v1, v2, v3, v4, v5, v6 { self.compile(v0, loss: v1, metrics: v2, loss_weights: v3, sample_weight_mode: v4, weighted_metrics: v5, target_tensors: v6) }
    ret.(val0, val1, val2, val3, val4, val5, val6)
  end
  
  def fit(val0, val1, val2 = 0, val3 = nil, val4 = 1, val5 = 1, val6 = nil, val7 = 0.0, val8 = nil, val9 = true, val10 nil, val11 = nil, val12 = 0, val13 = nil, val4 = nil)
    ret = -> v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14 { self.fit(v0, v1, batch_size: v3, epochs: v4, verbose: v5, callbacks: v6, validation_split: v7, validation_data: v8, shuffle: v9, class_weight: v10, sample_weight: v11, initial_epoch: v12, steps_per_epoch: v13, validation_steps: v14) }
    ret.(val0, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14)
  end
end
