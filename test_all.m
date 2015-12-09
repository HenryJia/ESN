#{

clear

leaking_rate = 0.0001
spectral_radius = 1
density = 0.01
reservoir_size = 1000
in_dim = 5 + 1;
#W_in_radius = sqrt(6 / (reservoir_size + in_dim))
W_in_radius = 1

Symbols = textread("symbols_clean.csv", '%s');

[W_in, W] = gen_reservoir(reservoir_size, in_dim, W_in_radius, density, spectral_radius);

for i = 1:size(Symbols, 1)
  filename{i} = char(strcat("stock_data_clean/", Symbols(i)));
  printf(strcat(filename{i}, '\n'));
  fflush(stdout);
  x{i} = load(filename{i})(1:end-200,:);
  x_val{i} = load(filename{i})(end-199:end,:);
  [final_internal_state{i}, W_out{i}, predicted{i}] = esn(x{i}, W_in, W, leaking_rate);
  train_error{i} = get_directional_accuracy(x{i}(51:end, 4), predicted{i});
  [final_internal_state_val{i}, predicted_val{i}] = esn_generate(x_val{i}(1:end - 1, :), final_internal_state{i}, W_out{i}, W_in, W, leaking_rate);
  validation_error{i} = get_directional_accuracy(x_val{i}, predicted_val{i});
  mse{i} = mean(abs(predicted{i} ./ x{i}(52:end, 4) - 1));
  validation_mse{i} = mean(abs(predicted_val{i} ./ x_val{i}(2:end, 4) - 1));
end

overall_train_error = mean(cell2mat(train_error))
overall_validation_error = mean(cell2mat(validation_error))
overall_train_mse = mean(cell2mat(mse))
overall_validation_mse = mean(cell2mat(validation_mse))
hist(cell2mat(validation_error), 10)

for i = 1:size(Symbols, 1)
  filename{i} = char(strcat("stock_data_clean/", Symbols(i)));
  printf(strcat(filename{i}, '\n'));
  fflush(stdout);
  x{i} = load(filename{i})(1:end-200,:);
  x_val{i} = load(filename{i})(end-199:end,:);
  [x_norm{i}, mu{i}, sigma{i}] = featureNormalize(x{i});
  x_val_norm{i} = featureNormalizeKnown(x_val{i}, mu{i}, sigma{i});
  [final_internal_state{i}, W_out{i}, W_in, W, predicted{i}] = esn(x_norm{i}, W_in, W, leaking_rate);
  train_error{i} = get_directional_accuracy(x_norm{i}(51:end, 4), predicted{i});
  [final_internal_state_val{i}, predicted_val{i}] = esn_generate(x_val_norm{i}(1:end - 1, :), final_internal_state{i}, W_out{i}, W_in, W, leaking_rate);
  validation_error{i} = get_directional_accuracy(x_val{i}, predicted_val{i});
  mse{i} = mean(abs(featureRecover(predicted{i}, mu{i}, sigma{i}) ./ x{i}(52:end, 4) - 1));
  validation_mse{i} = mean(abs(featureRecover(predicted_val{i}, mu{i}, sigma{i}) ./ x_val{i}(2:end, 4) - 1));
end


for i = 1:size(Symbols, 1)
  [final_internal_state_val2{i}, predicted_val2{i}] = esn_generate(x_val{i}(1:end - 1, :), final_internal_state_val{i}(7:end), W_out{i}, W_in, W, leaking_rate);
  validation_error2{i} = get_directional_accuracy(x_val{i}, predicted_val2{i});
  validation_mse2{i} = mean(abs(predicted_val2{i} ./ x_val{i}(2:end, 4) - 1));
end
overall_validation_error2 = mean(cell2mat(validation_error2))
overall_validation_mse2 = mean(cell2mat(validation_mse2))
#}