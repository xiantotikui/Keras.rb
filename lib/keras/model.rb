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
end
