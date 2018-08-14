module Keras::Noise
  extend self
  pyfrom 'keras.layers', import: 'GaussianNoise'
  pyfrom 'keras.layers', import: 'GaussianDropout'
  pyfrom 'keras.layers', import: 'AlphaDropout'

  def gaussian_noise(stddev, **args)
    GaussianNoise.new(stddev, **args)
  end

  def gaussian_dropout(rate, **args)
    GaussianDropout.new(rate, **args)
  end

  def alpha_dropout(rate, **args)
    AlphaDropout.new(rate, **args)
  end
end
