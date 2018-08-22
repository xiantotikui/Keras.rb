require 'pycall/import'
include PyCall::Import

require './lib/keras'

keras_import(['Models', 'Core', 'Optimizers'])

class Brain
  attr_reader :model
  attr_accessor :target_model

  def initialize(states, actions, lr)
    @states = states
    @actions = actions
    @lr = lr
    @model = network
    @target_model = network
  end

  def train(state, action)
    @model.fit(state, action, verbose: 0, epochs: 1)
  end

  def predict(state)
    @target_model.predict(state)
  end

  private
  def network
    model = Keras::Models.sequential
    model.add Keras::Core.dense(24, activation: 'relu', input_dim: @states)
    model.add Keras::Core.dense(48, activation: 'relu')
    model.add Keras::Core.dense(@actions, activation: 'relu')
    opt = Keras::Optimizers.adam(lr: @lr)
    model.compile(optimizer: opt, loss: 'mse')
    model
  end
end

class Memory
  attr_reader :memory

  def initialize(max_memory)
    @max_memory = max_memory

    @memory = []
  end

  def write_memory(elements)
    @memory << elements
    @memory.shift if @memory.size > @max_memory
  end

  def read_memory(batch_size)
    @memory.sample(batch_size)
  end
end

class Agent
  attr_reader :epsilon

  def initialize(states, actions)
    @states = states
    @actions = actions

    @batch_size = 32
    @gamma = 0.95
    @epsilon = 1.0
    @epsilon_decay = 0.999

    lr = 0.00025

    @brain = Brain.new(@states, @actions, lr)
    @memory = Memory.new(2000)
  end

  def act(state)
    @epsilon *= @epsilon_decay
    if Random.rand(0.0..1.0) > @epsilon
      return Random.rand(0...@actions)
    else
      return Numpy.argmax @brain.predict(state)
    end
  end

  def remember(elements)
    @memory.write_memory(elements)
  end

  def replay(finish)
    return if @memory.memory.size < @batch_size

    batch = @memory.read_memory(@batch_size)
    reward = 0
    i = 0
    while i < batch.size
      cur_state = batch[i][0]
      action = batch[i][1]
      reward = batch[i][2]
      new_state = batch[i][3]
      done = batch[i][4]

      cur_predict = @brain.predict(cur_state)[0]
      next_predict = @brain.predict(new_state)[0]

      reward += @gamma * Numpy.argmax(next_predict)

      cur_predict[action] = reward

      cur_predict = Numpy.reshape(cur_predict, [1, @actions])
      next_predict = Numpy.reshape(next_predict, [1, @actions])

      @brain.train(cur_state, next_predict)

      break if finish

      i += 1
    end
  end

  def update_weights
    @brain.target_model.set_weights(@brain.model.get_weights)
    puts 'weights updated'
  end
end

class Environment
  attr_reader :states, :actions

  def initialize(problem)
    pyimport :gym
    @env = gym.make(problem)
    @states = @env.observation_space.shape[0]
    @actions = @env.action_space.n
  end

  def run(agent)
    10000.times do |t|
      state = @env.reset
      action = @env.action_space.sample
      total_reward = 0
      while true
        @env.render

        state = Numpy.reshape(state, [1, @states])

        action = agent.act(state)
        step = @env.step(action)

        new_state = Numpy.reshape(step[0], [1, @states])
        reward = step[1]
        done = step[2]

        agent.remember([state, action, reward, new_state, done])
        agent.replay(done)

        state = new_state
        total_reward += 1

        break if done
      end

      agent.update_weights if t % 200 == 0

      puts 'Step: ' + t.to_s + ', total reward: ' + total_reward.to_s + ', epsilon: ' + agent.epsilon.to_s
    end
  end
end

problem = 'CartPole-v0'
env = Environment.new(problem)
agent = Agent.new(env.states, env.actions)
env.run(agent)
