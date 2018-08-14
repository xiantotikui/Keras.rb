module Callbacks
  extend self

  def callback
    pyfrom 'keras.callbacks', import: 'Callback'
    Callback.new
  end

  def base_logger(stateful_metrics: nil, **args)
    pyfrom 'keras.callbacks', import: 'BaseLogger'
    BaseLogger.new(stateful_metrics: stateful_metrics, **args)
  end

  def terminate_on_nan
    pyfrom 'keras.callbacks', import: 'TerminateOnNaN'
    TerminateOnNaN.new
  end

  def callback(count_mode: 'samples', **args)
    pyfrom 'keras.callbacks', import: 'ProgbarLogger'
    ProgbarLogger.new(count_mode: count_mode, **args)
  end

  def history
    pyfrom 'keras.callbacks', import: 'History'
    History.new
  end

  def model_checkpoint(filepath, **args)
    pyfrom 'keras.callbacks', import: 'ModelCheckpoint'
    ModelCheckpoint.new(filepath, **args)
  end

  def early_stopping(monitor: 'val_loss', **args)
    pyfrom 'keras.callbacks', import: 'EarlyStopping'
    EarlyStopping.new(monitor: monitor, **args)
  end

  def remote_monitor(root: 'http://localhost:9000', **args)
    pyfrom 'keras.callbacks', import: 'RemoteMonitor'
    RemoteMonitor.new(root: root, **args)
  end

  def learning_rate_scheduler(schedule, **args)
    pyfrom 'keras.callbacks', import: 'LearningRateScheduler'
    LearningRateScheduler.new(schedule, **args)
  end

  def tensor_board(log_dir: './logs', **args)
    pyfrom 'keras.callbacks', import: 'TensorBoard'
    TensorBoard.new(log_dir: log_dir, **args)
  end

  def reducelr_on_plateau(monitor: 'val_loss', **args)
    pyfrom 'keras.callbacks', import: 'ReduceLROnPlateau'
    ReduceLROnPlateau.new(monitor: monitor, **args)
  end

  def csv_logger(filename, **args)
    pyfrom 'keras.callbacks', import: 'CSVLogger'
    CSVLogger.new(filename, **args)
  end
end
