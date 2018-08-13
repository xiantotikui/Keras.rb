module Keras::Embedding
  extend self
  pyfrom 'keras.layers', import: 'Embedding'

  def embedding(input_dim, output_dim, **args)
    Embedding.new(input_dim, output_dim, **args)
  end
end
