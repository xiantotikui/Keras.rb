require 'open-uri'
require './lib/keras'

python_version('~/miniconda3/bin/python')

keras_import(['Datasets', 'Utils', 'Model', 'Recurrent', 'Core', 'Optimizers'])

text_name = 'pan-tadeusz.txt'

File.open(text_name, "wb") do |file|
  file.write open('https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt').read
end

text = File.read(text_name).downcase.chars.to_a
chars_array = text.uniq

char_indices = Hash.new
indices_char = Hash.new

chars_array.each.with_index do |ch, idx|
  char_indices[ch] = idx
end

chars_array.each.with_index do |ch, idx|
  indices_char[idx] = ch
end

maxlen = 40
step = 3
sentences = []
next_chars = []

i = 0
while i < text.size - maxlen
  sentences << text[i...(i + maxlen)]
  next_chars << text[i + maxlen]
  i += step
end

x = []
y = []

i = 0
while i < 20000
  x[i] = []
  j = 0
  while j < sentences[i].size
    x[i][j] = Array.new(indices_char.size).fill(0)
    x[i][j][char_indices[sentences[i][j]]] = 1
    j += 1
  end
  y[i] = Array.new(indices_char.size).fill(0)
  y[i][char_indices[next_chars[i]]] = 1
  i += 1
end

model = Keras::Model.sequential
model.add(Keras::Recurrent.lstm(128, input_shape: [maxlen, chars_array.size]))
model.add(Keras::Core.dense(chars_array.size))
model.add(Keras::Core.activation('softmax'))

optimizer = Keras::Optimizers.rmsprop(lr: 0.01)
model.compile(loss: 'categorical_crossentropy', optimizer: optimizer)
model.fit(Numpy.array(x), Numpy.array(y), batch_size: 128, epochs: 60)

start_index = Random.rand(0...(text.size - maxlen))
sentence = text[start_index...(start_index + maxlen)]
x_pred = []
generated = ''
i = 0
while i < 400
  x_pred[0] = []
  j = 0
  while j < maxlen
    x_pred[0][j] = Array.new(indices_char.size).fill(0)
    x_pred[0][j][char_indices[sentence[j]]] = 1
    j += 1
  end
  preds = model.predict(Numpy.array(x_pred), verbose: 0)[0]
  preds = Numpy.asarray(preds).astype('float64')
  preds = Numpy.log(preds)
  exp_preds = Numpy.exp(preds)
  preds = exp_preds / Numpy.sum(exp_preds)
  probas = Numpy.multinomial(1, preds, 1)
  next_index = Numpy.argmax(probas)
  next_char = indices_char[next_index.to_i]
  sentence = sentence[1..maxlen] + [next_char]
  generated << next_char
  i += 1
end

puts generated
