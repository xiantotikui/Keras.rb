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
end
