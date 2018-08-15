require 'pycall/import'
include PyCall::Import

pyimport :gym

require './lib/keras'

python_version('~/miniconda3/bin/python')

keras_import(['Model', 'Convolutional', 'Core', 'Optimizers'])

class DQNAgent
  attr_accessor :epsilon
  attr_reader :memory

  def initialize(state_size, action_size)
    @state_size = state_size
    @action_size = action_size
    @memory = []
    @gamma = 0.95
    @tau = 0.5
    @epsilon = 1.0
    @epsilon_min = 0.01
    @epsilon_decay = 0.995
    @learning_rate = 0.001
    @model = build_model
  end

  def build_model
    model = Keras::Model.sequential
    model.add(Keras::Core.dense(64, input_dim: @state_size, activation: 'relu'))
    model.add(Keras::Core.dense(@action_size, activation: 'linear'))
    model.compile(loss: 'mse', optimizer: Keras::Optimizers.adam(lr: @learning_rate))
    model
  end

  def remember(state, action, reward, next_state, done, last_reward)
    @memory << [state, action, reward, next_state, done, last_reward]
  end

  def act(state)
    if Random.rand(0.0..1.0) <= @epsilon
      ret = Random.rand(0...@action_size)
    else
      ret = Numpy.argmax(@model.predict(state))
    end
    ret
  end

  def sigmoid(x)
    1.0 / (1.0 + Math.exp(-x))
  end

  def replay(batch_size)
    minibatch = @memory.last(batch_size)
    i = 0
    while i < minibatch.size
      target = @memory[i][2]
      tmp = (@memory[i][2] + @gamma * Numpy.amax(@model.predict(@memory[i][3]))) + @tau * @memory[i][5]
      if !@memory[i][4]
        target = sigmoid(tmp.to_f)
      else
        target = -1.0 * sigmoid(tmp.to_f)
      end
      target_f = @model.predict(@memory[i][0])
      if @memory[i][1].zero?
        target_f[0][0] = target
        target_f[0][1] = 0.0
      else
        target_f[0][0] = 0.0
        target_f[0][1] = target
      end
      @model.fit(@memory[i][0], target_f, epochs: 1, verbose: 0)
      if @epsilon > @epsilon_min
        @epsilon *= @epsilon_decay
      end
      i += 1
    end
  end

  def load(name)
    @model.load_weights(name)
  end

  def save(name)
    @model.load_weights(name)
  end
end

episodes = 1000

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent.new(state_size, action_size)
done = false
batch_size = 32

episodes.times do |e|
  state = env.reset()
  agent.epsilon = 1.0
  last_reward = []
  state = Numpy.reshape(state, [1, state_size])
  500.times do |t|
    env.render()
    action = agent.act(state)
    tmp = env.step(action)
    next_state = tmp[0]
    reward = tmp[1]
    done = tmp[2]
    next_state = Numpy.reshape(next_state, [1, state_size])
    last_reward = t
    agent.remember(state, action, reward, next_state, done, last_reward)
    state = next_state
    if done
      puts("episode: #{e}/#{episodes}, score: #{t}, e: #{agent.epsilon}")
      break
    end
    if agent.memory.size > batch_size
      agent.replay(batch_size)
    end
  end
end
