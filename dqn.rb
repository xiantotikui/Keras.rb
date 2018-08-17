require 'pycall/import'
include PyCall::Import

require './lib/keras'

python_version('~/miniconda3/bin/python')

keras_import(['Model', 'Convolutional', 'Core', 'Optimizers'])

class Environment
  attr_reader :state, :action

  def initialize
    pyimport :gym
    @env = gym.make 'CartPole-v0'
    states = @env.observation_space.shape[0]
    actions = @env.action_space.n
    @agent = Agent.new(states, actions)
  end

  def run
    @steps = 0
    @agent.epsilon = 1.0
    state = @env.reset
    done = false
    while !done
      @env.render
      action = @agent.act(state)
      array = @env.step(action)
      next_state = array[0]
      reward = array[1]
      done = array[2]

      @agent.observe [state, action, reward, next_state]
      @agent.replay

      state = next_state
      @steps += 1
    end

    puts 'Total reward: ' + @steps.to_s
  end
end

class Agent
  attr_writer :epsilon
  attr_reader :state

  def initialize(states, actions)
    @states = states
    @actions = actions
    @batch_size = 64

    @epsilon = 1.0
    @min_epsilon = 0.01
    @epsilon_decay = 0.995

    @gamma = 0.95

    @brain = Brain.new(@states, @actions, @batch_size)
  end

  def act(state)
    if Random.rand(0.0..1.0) <= @epsilon
      @state = Random.rand(0...@actions)
    else
      @state = Numpy.argmax @brain.predict state
    end
  end

  def observe(sample)
    @brain.write_memory sample
    @epsilon *= @epsilon_decay
  end

  def replay
    batch = @brain.read_memory
    zero_state = Numpy.zeros(@states)

    batch.size.times do |t|
      state = batch[t][0]
      action = batch[t][1]
      reward = batch[t][2]
      if batch[t][3].nil?
        next_state = zero_state
      else
        next_state = batch[t][3]
      end
      predicted_state = @brain.predict(state)
      predicted_next_state = @brain.predict(next_state)

      target = predicted_state

      if next_state.nil?
        target[action] = reward
      else
        target[action] = reward + @gamma * Numpy.amax(predicted_next_state)
      end

      @brain.train(Numpy.reshape(state, [1, @states]), Numpy.reshape(target, [1, @actions]))
    end
  end
end

class Brain
  def initialize(states, actions, batch_size)
    @batch_size = batch_size

    @states = states
    @actions = actions

    @memory = Memory.new(200)
    @model = build_model
  end

  def build_model
    model = Keras::Model.sequential
    model.add Keras::Core.dense 64, activation: 'relu', input_dim: @states
    model.add Keras::Core.dense @actions, activation: 'linear'
    model.compile loss: 'mse', optimizer: Keras::Optimizers.adam(lr: 0.0005, clipvalue: 1.0)
    model
  end

  def predict(state)
    (@model.predict Numpy.reshape(state, [1, @states])).flatten
  end

  def train(x, y)
    @model.fit x, y, batch_size: @batch_size, epochs: 1, verbose: 0
  end

  def read_memory
    @memory.sampl @batch_size
  end

  def write_memory(sample)
    @memory.add sample
  end
end

class Memory
  def initialize(capacity)
    @capacity = capacity
    @memory = []
  end

  def add(array)
    @memory << array
    @memory.shift if @memory.size > @capacity
  end

  def sampl(n)
    @memory.sample(n)
  end
end

environment = Environment.new

i = 0
while i < 1000
  puts 'Step: ' + i.to_s
  environment.run
  i += 1
end
