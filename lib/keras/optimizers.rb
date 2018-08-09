module Optimizers
  pyimport :keras
  extend self

  def sgd(val0 = 0.01, val1 = 0.0, val2 = 0.0, val3 = false)
    ret = -> v0, v1, v2, v3 { keras.optimizers.SGD.new(lr: v0, momentum: v1, decay: v2, nesterov: v3) }
    ret.(val0, val1, val2, val3)
  end
end
