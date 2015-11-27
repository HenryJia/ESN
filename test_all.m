#{

leaking_rate = 0.01
spectral_radius = 1
reservoir_size = 1000;
in_dim = 5 + 1;

Symbols = textread("symbols_clean.csv", '%s');

[W_in, W] = gen_reservoir(reservoir_size, in_dim, spectral_radius);

for i = 1:size(Symbols, 1)
  filename{i} = char(strcat("stock_data_clean/", Symbols(i)));
  x{i} = load(filename{i})(1:end-200,:);
  x_val{i} = load(filename{i})(end-199:end,:);
  [final_internal_state{i}, W_out{i}, W_in, W, predicted{i}] = esn(x{i}, W_in, W, leaking_rate);
  train_error{i} = get_directional_accuracy(x{i}(51:end, 3), predicted{i});
  [final_internal_state_val{i}, predicted_val{i}] = esn_generate(x_val{i}(1:end - 1, :), final_internal_state{i}, W_out{i}, W_in, W, leaking_rate);
  validation_error{i} = get_directional_accuracy(x_val{i}, predicted_val{i});
  mse{i} = mean(abs(predicted{i} ./ x{i}(52:end, 3) - 1));
  validation_mse{i} = mean(abs(predicted_val{i} ./ x_val{i}(2:end, 3) - 1));
end

overall_train_error = mean(cell2mat(train_error))
overall_validation_error = mean(cell2mat(validation_error))
overall_train_mse = mean(cell2mat(mse))
overall_validation_mse = mean(cell2mat(validation_mse))
hist(cell2mat(validation_error), 10)

#}