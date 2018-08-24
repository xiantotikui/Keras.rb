require 'pycall/import'
include PyCall::Import

require './lib/keras'

keras_import(['Models', 'Core', 'Optimizers'])

class Brain
  attr_reader :online_value, :online_advantage
  attr_accessor :target_value, :target_advantage

  def initialize(states, actions, lr)
    @states = states
    @actions = actions
    @lr = lr
    @online_value = value_network
    @target_value = value_network

    @online_advantage = advantage_network
    @target_advantage = advantage_network
  end

  def train_value(state, action)
    @online_value.fit(state, action, verbose: 0, epochs: 1)
  end

  def predict_value(state, target: false)
    if target
      @target_value.predict(state)
    else
      @online_value.predict(state)
    end
  end

  def predict_advantage(sa)
    @target_advantage.predict(sa)
  end

  def train_advantage(sa, action)
    @online_advantage.fit(sa, action, verbose: 0, epochs: 1)
  end

  private
  def value_network
    model = Keras::Models.sequential
    model.add Keras::Core.dense(24, activation: 'relu', input_dim: @states)
    model.add Keras::Core.dense(48, activation: 'relu')
    model.add Keras::Core.dense(@actions, activation: 'linear')
    opt = Keras::Optimizers.adam(lr: @lr)
    model.compile(optimizer: opt, loss: 'huber')
    model
  end

  def advantage_network
    model = Keras::Models.sequential
    model.add Keras::Core.dense(24, activation: 'relu', input_dim: @states + @actions)
    model.add Keras::Core.dense(48, activation: 'relu')
    model.add Keras::Core.dense(@actions, activation: 'linear')
    opt = Keras::Optimizers.adam(lr: @lr)
    model.compile(optimizer: opt, loss: 'huber')
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
  attr_writer :update

  def initialize(states, actions)
    @states = states
    @actions = actions

    @update = false

    @batch_size = 32
    @gamma = 0.95
    @epsilon = 1.0
    @epsilon_decay = 0.995
    @min_epsilon = 0.1

    lr = 0.0005

    @brain = Brain.new(@states, @actions, lr)
    @memory = Memory.new(400)
  end

  def act(state)
    @epsilon *= @epsilon_decay
    @epsilon = @min_epsilon if @epsilon < @min_epsilon
    if Random.rand(0.0..1.0) > @epsilon
      ret = Random.rand(0...@actions)
    else
      ret = Numpy.argmax @brain.predict_value(state, target: false)[0]
    end
    if @update
      update_val_weights
      update_adv_weights
      p 'weights updated!'
    end
    ret
  end

  def remember(elements)
    @memory.write_memory(elements)
  end

  def replay
    return if @memory.memory.size < @batch_size

    batch = @memory.read_memory(@batch_size)

    cur_state = []
    cur_online_val = []
    cur_online_adv = []
    next_online_adv = []
    cur_adv = []

    i = 0
    while i < batch.size
      cur_state[i] = batch[i][0]
      action = batch[i][1]
      next_state = batch[i][2]
      done_sample = batch[i][3]

      cur_online_val[i] = @brain.predict_value(cur_state[i], target: false)[0]
      next_online_val = @brain.predict_value(next_state, target: false)[0]
      next_target_val = @brain.predict_value(next_state, target: true)[0]

      prev_adv = Numpy.append(cur_state[i], cur_online_val[i])
      prev_adv = Numpy.reshape(prev_adv, [1, @states + @actions])

      if done_sample
        cur_online_val[i][action] = 0.0
      else
        cur_online_val[i][action] = @gamma * next_target_val[Numpy.argmax(next_online_val)]
      end

      cur_adv[i] = Numpy.append(cur_state[i], cur_online_val[i])
      cur_adv[i] = Numpy.reshape(cur_adv[i], [1, @states + @actions])

      cur_online_adv[i] = @brain.predict_advantage(prev_adv)
      next_online_adv[i] = @brain.predict_advantage(cur_adv[i])

      i += 1
    end

    avg_adv = next_online_adv.inject(:+) / batch.size

    i = 0
    while i < batch.size
      cur_online_val[i] += cur_online_adv[i] - avg_adv

      cur_online_val[i] = Numpy.reshape(cur_online_val[i], [1, @actions])
      @brain.train_value(cur_state[i], cur_online_val[i])

      @brain.train_advantage(cur_adv[i], cur_online_val[i])

      i += 1
    end
  end

  def update_val_weights
    @brain.target_value.set_weights(@brain.online_value.get_weights)
  end

  def update_adv_weights
    @brain.target_advantage.set_weights(@brain.online_advantage.get_weights)
  end
end

class Environment
  attr_reader :states, :actions

  def initialize(problem)
    pyimport :gym
    @env = gym.make(problem)
    @states = @env.observation_space.shape[0]
    @actions = @env.action_space.n

    @update_rate = 100
  end

  def run(agent)
    10000.times do |time|
      state = @env.reset
      action = @env.action_space.sample
      total_reward = 0
      if time % @update_rate == 0
        agent.update = true
      end
      while true
        @env.render

        state = Numpy.reshape(state, [1, @states])

        action = agent.act(state)
        agent.update = false

        step = @env.step(action)

        next_state = Numpy.reshape(step[0], [1, @states])
        reward = step[1]
        done = step[2]

        agent.remember([state, action, next_state, done])
        agent.replay

        state = next_state
        total_reward += 1

        break if done
      end

      puts 'Step: ' + time.to_s + ', total reward: ' + total_reward.to_s + ', epsilon: ' + agent.epsilon.to_s
    end
  end
end

problem = 'CartPole-v0'
env = Environment.new(problem)
agent = Agent.new(env.states, env.actions)
env.run(agent)
