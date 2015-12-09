function [final_internal_state, W_out, predicted] = esn(x, W_in, W, leaking_rate = 1)

#{
To run:
spectral_radius = 1
density = 0.01
reservoir_size = 1000
in_dim = 5 + 1;
W_in_radius = 1
[W_in, W] = gen_reservoir(reservoir_size, in_dim, W_in_radius, density, spectral_radius);

W_in *= 1e-10;

leaking_rate = 0.0001
x = load("stock_data_clean/SBRY_LDX.TXT" )(1:end-200,:);
x_val = load("stock_data_clean/SBRY_LDX.TXT" )(end-199:end,:);
tic
[final_internal_state, W_out, predicted] = esn(x, W_in, W, leaking_rate);
toc
get_directional_accuracy(x(51:end, 4), predicted)
[final_internal_state_val, predicted_val] = esn_generate(x_val(1:end - 1, :), final_internal_state, W_out, W_in, W, leaking_rate);
get_directional_accuracy(x_val, predicted_val)
mean(abs(predicted ./ x(52:end, 4) - 1))
mean(abs(predicted_val ./ x_val(2:end, 4) - 1))
figure(1)
hold off
plot(1:size(x_val, 1)-1, x_val(2:end, 4), "b", "linewidth", 1)
hold on
plot(1:size(x_val, 1)-1, predicted_val, "r", "linewidth", 1)
hold off
figure(2)
hist (abs(predicted_val ./ x_val(2:end, 4) - 1), 10, norm=true);

# Normalized version

spectral_radius = 0.3
density = 0.95
reservoir_size = 100
in_dim = 5 + 4 * 4 + 1;
#W_in_radius = sqrt(6 / (reservoir_size + in_dim))
#W_in_radius = 0.5
W_in_radius = 1 / sqrt(in_dim)
[W_in, W] = gen_reservoir(reservoir_size, in_dim, W_in_radius, density, spectral_radius);

leaking_rate = 1
x = load("stock_data_clean/KAZ_LDX.TXT" )(1:end-200,:);
x_val = load("stock_data_clean/KAZ_LDX.TXT" )(end-199:end,:);
x = [x, movavg(x, 5, 5, 0)(:, 1:4), movavg(x, 10, 10, 0)(:, 1:4), movavg(x, 15, 15, 0)(:, 1:4), movavg(x, 20, 20, 0)(:, 1:4)];
x_val = [x_val, movavg(x_val, 5, 5, 0)(:, 1:4), movavg(x_val, 10, 10, 0)(:, 1:4), movavg(x_val, 15, 15, 0)(:, 1:4), movavg(x_val, 20, 20, 0)(:, 1:4)];
[x_norm, mu, sigma] = featureNormalize(x);
x_val_norm = featureNormalizeKnown(x_val, mu, sigma);
[final_internal_state, W_out,  predicted] = esn(x_norm, W_in, W, leaking_rate);
get_directional_accuracy(x_norm(51:end, 4), predicted)
[final_internal_state_val, predicted_val] = esn_generate(x_val_norm(1:end - 1, :), final_internal_state, W_out, W_in, W, leaking_rate);
get_directional_accuracy(x_val_norm, predicted_val)
mean(abs(featureRecover(predicted, mu, sigma) ./ x(52:end, 4) - 1))
mean(abs(featureRecover(predicted_val, mu, sigma) - x_val(2:end, 4) - 1))
figure(1)
hold off
plot(1:size(x_val, 1)-1, x_val_norm(2:end, 4), "b", "linewidth", 1)
hold on
plot(1:size(x_val, 1)-1, predicted_val, "r", "linewidth", 1)
hold off
figure(2)
hist (abs(featureRecover(predicted_val, mu, sigma) ./ x_val(2:end, 4) - 1), 10, norm=true);

xy = zeros(50, 2);

for i = 1:5
  for j = 1:10
  xy(i * 10 + j, :) = [i, j];
  end
end

#}

  #lambda = 5;
  lambda = 0;
  flush_length = 50;
  in_dim = size(x, 2) + 1;
  reservoir_size = size(W, 1);

  internal_state = rescale(rand(reservoir_size, 1));
  final_internal_state = zeros(size(x, 1) - flush_length, reservoir_size + in_dim);
  update = zeros(reservoir_size, 1);  

  # Run the reservoir
  #max_size = 2050;
  #if (size(x, 1) < 2050)
  #  max_size = size(x, 1);
  #end
  #for i = 1:max_size
  #up = tanh(W_in * [1; x(1, :)'] + W * internal_state)
  for i = 1:size(x, 1)
    update = tanh(W_in * [1; x(i, :)'] + W * internal_state);
    internal_state = (1 - leaking_rate) * internal_state + leaking_rate * update;

    if (i > flush_length)
      final_internal_state(i - flush_length, :) = [1, x(i, :), internal_state'];
    end
  end

  targets = x(flush_length+2:end, 4);
  data = final_internal_state(1:end - 1, :);

  W_out = pinv(data' * data + lambda * eye(in_dim + reservoir_size)) * data' * targets ;

  predicted = data * W_out;
  figure(10)
  plot(final_internal_state(end-50:end, randi([in_dim + 1, 100 + in_dim])))
  final_internal_state = final_internal_state(end, in_dim + 1:end);
end