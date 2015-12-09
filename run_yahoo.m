function [final_internal_state, predictions, x] = run_yahoo(leaking_rate = 0.0001)
  initial_internal_state = load("final_internal_state.bin").final_internal_state;
  W_in = load("W_in.bin").W_in;
  W = load("W.bin").W;
  W_out = load("W_out.bin").W_out;

  Symbols = char(textread("symbols_clean_yahoo.csv", '%s'));

  for i=1:size(Symbols, 1)
    x{i} = fetch_yahoo_current(Symbols(i, :));
    printf(strcat(Symbols(i, :), '\n'));
    fflush(stdout);
    [internal_state, predicted] = esn_generate(x{i}, initial_internal_state{i}, W_out{i}, W_in, W, leaking_rate);
    final_internal_state{i} = internal_state;
    predictions{i} = predicted;
  end

  #save "final_internal_state.bin" final_internal_state
  #x_mat = reshape(cell2mat(x), [5, 148])';
  #validation_error = load("validation_error.bin").validation_error;
  #validation_mse = load("validation_mse.bin").validation_mse;

  #{
  [final_internal_state, predictions, x] = run_yahoo(leaking_rate = 0.0001);
  validation_error = load("validation_error.bin").validation_error
  validation_error = load("validation_error.bin").validation_error;
  validation_mse = load("validation_mse.bin").validation_mse;
  ones = (1:148)';
  format short g
  prediction_mat = cell2mat(predictions)';
  x_mat = reshape(cell2mat(x), [5, 148])';
  percentage = (prediction_mat ./ x_mat(:, 4) - 1) * 100;
  [ones, percentage, cell2mat(validation_error)', cell2mat(validation_mse)' * 100]
  #}
end