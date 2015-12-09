function [final_internal_state, W_out, predicted] = esn_classify(x, W_in, W, leaking_rate = 1)

#{
To run:
spectral_radius = 1
density = 0.01
reservoir_size = 1000
in_dim = 5 + 1;
W_in_radius = 1
[W_in, W] = gen_reservoir(reservoir_size, in_dim, W_in_radius, density, spectral_radius);

leaking_rate = 0.0001
x = load("stock_data_clean/SBRY_LDX.TXT" )(1:end-200,:);
x_val = load("stock_data_clean/SBRY_LDX.TXT" )(end-199:end,:);
tic
[final_internal_state, W_out, predicted] = esn_classify(x, W_in, W, leaking_rate);
toc
get_directional_accuracy_classify(x(51:end, 4), predicted)
[final_internal_state_val, predicted_val] = esn_generate(x_val(1:end - 1, :), final_internal_state, W_out, W_in, W, leaking_rate);
get_directional_accuracy(x_val, predicted_val)
mean(abs(predicted ./ x(52:end, 4) - 1))
mean(abs(predicted_val ./ x_val(2:end, 4) - 1))
figure(1)
hold off
plot(1:size(x_val, 1)-1, x_val(2:end, 4), "bo", "linewidth", 1)
hold on
plot(1:size(x_val, 1)-1, predicted_val, "r+", "linewidth", 1)
hold off
figure(2)
hist (abs(predicted_val ./ x_val(2:end, 4) - 1), 10, norm=true);

# Normalized version

spectral_radius = 1
density = 0.01
reservoir_size = 1000
in_dim = 5 + 1;
W_in_radius = sqrt(6 / (reservoir_size + in_dim))
#W_in_radius = 0.5
[W_in, W] = gen_reservoir(reservoir_size, in_dim, W_in_radius, density, spectral_radius);

leaking_rate = 0.005
x = load("stock_data_clean/KAZ_LDX.TXT" )(1:end-200,:);
x_val = load("stock_data_clean/KAZ_LDX.TXT" )(end-199:end,:);
[x_norm, mu, sigma] = featureNormalize(x);
x_val_norm = featureNormalizeKnown(x_val, mu, sigma);
[final_internal_state, W_out,  predicted] = esn_classify(x_norm, W_in, W, leaking_rate);
get_directional_accuracy_classify(x(51:end, 4), predicted)
[final_internal_state_val, predicted_val] = esn_generate(x_val_norm(1:end - 1, :), final_internal_state, W_out, W_in, W, leaking_rate);
(sum((x_val_norm(2:end, 4) >= 0) .* (predicted_val >= 0)) + sum((x_val_norm(2:end, 4) < 0) .* (predicted_val < 0))) / size(predicted_val, 1)
mean(abs(predicted - x_norm(52:end, 4)))
mean(abs(predicted_val - x_val_norm(2:end, 4)))
figure(1)
hold off
plot(1:size(x_val, 1)-1, x_val(2:end, 4), "bo", "linewidth", 1)
hold on
plot(1:size(x_val, 1)-1, predicted_val, "r+", "linewidth", 1)
hold off
figure(2)
hist (abs(featureRecover(predicted_val, mu, sigma) ./ x_val(2:end, 4) - 1), 10, norm=true);
#}

  flush_length = 50;
  learning_rate = 0.01;
  in_dim = size(x, 2) + 1;
  reservoir_size = size(W, 1);

  internal_state = zeros(reservoir_size, 1);
  final_internal_state = zeros(size(x, 1) - flush_length, reservoir_size + in_dim);
  update = zeros(reservoir_size, 1);  

  # Run the reservoir
  for i = 1:size(x, 1)
    update = hard_tanh(W_in * [1; x(i, :)'] + W * internal_state);
    internal_state = (1 - leaking_rate) * internal_state + leaking_rate * update;

    if (i > flush_length)
      final_internal_state(i - flush_length, :) = [1, x(i, :), internal_state'];
    end
  end

  targets = (x(flush_length+2:end, 4) ./ x(flush_length+1:end-1, 4) - 1) >= 0;
  data = final_internal_state(1:end-1, :);

  reservoir_size + in_dim

  W_out = zeros(reservoir_size + in_dim, 1);
  for i=1:1000
  grad = (((sigmoid(data * W_out) - targets)' * data)/size(data, 1))';
  #J = (log(sigmoid(linear))' * y + log(1 - sigmoid(linear))' * (1 - y))/(-m)

  W_out = W_out - grad * learning_rate;
  end

  predicted = sigmoid(data * W_out);
  #figure(1)
  #plot(final_internal_state(:, randi([1, 1000])))
  final_internal_state = final_internal_state(end, 7:end);
end