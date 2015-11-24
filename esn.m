function [final_internal_state, W_out, W_in, W, predicted] = esn(x, leaking_rate = 1)

#{
To run:
leaking_rate = 0.1
x = load("stock_data_clean/KAZ_LDX.TXT" )(1:end-200,:);
x_val = load("stock_data_clean/KAZ_LDX.TXT" )(end-199:end,:);
[final_internal_state, W_out, W_in, W, predicted] = esn(x, leaking_rate);
get_directional_accuracy(x(51:end, 3), predicted)
[final_internal_state_val, predicted_val] = esn_generate(x_val(1:end - 1, :), final_internal_state, W_out, W_in, W, leaking_rate);
get_directional_accuracy(x_val, predicted_val)
mean(abs(predicted ./ x(52:end, 3) - 1))
mean(abs(predicted_val ./ x_val(2:end, 3) - 1))
figure(1)
hold off
plot(1:size(x_val, 1)-1, x_val(2:end, 3), "bo", "linewidth", 1)
hold on
plot(1:size(x_val, 1)-1, predicted_val, "r+", "linewidth", 1)
hold off
figure(2)
hist (abs(predicted_val ./ x_val(2:end, 3) - 1), 10, norm=true);

# Normalized version

leaking_rate = 0.1
x = load("stock_data_clean/KAZ_LDX.TXT" )(1:end-200,:);
x_val = load("stock_data_clean/KAZ_LDX.TXT" )(end-199:end,:);
[x_norm, mu, sigma] = featureNormalize(x);
x_val_norm = featureNormalizeKnown(x_val, mu, sigma);
[final_internal_state, W_out, W_in, W, predicted] = esn(x_norm, leaking_rate);
get_directional_accuracy(x_norm(51:end, 3), predicted)
[final_internal_state_val, predicted_val] = esn_generate(x_val_norm(1:end - 1, :), final_internal_state, W_out, W_in, W, leaking_rate);
get_directional_accuracy(x_val_norm, predicted_val)
mean(abs(featureRecover(predicted, mu, sigma) ./ x(52:end, 3) - 1))
mean(abs(featureRecover(predicted_val, mu, sigma) ./ x_val(2:end, 3) - 1))
figure(1)
hold off
%plot(1:size(x_val, 1)-1, x_val(2:end, 3), "bo", "linewidth", 1)
hold on
plot(1:size(x_val, 1)-1, featureRecover(predicted_val, mu, sigma), "r+", "linewidth", 1)
hold off
figure(2)
hist (abs(featureRecover(predicted_val, mu, sigma) ./ x_val(2:end, 3) - 1), 10, norm=true);
#}

  reservoir_size = 1000
  leaking_rate
  flush_length = 50
  in_dim = size(x, 2) + 1

  internal_state = zeros(reservoir_size, 1);
  final_internal_state = zeros(size(x, 1) - flush_length, reservoir_size + in_dim);
  update = zeros(reservoir_size, 1);  

  [W_in, W] = gen_reservoir(reservoir_size, in_dim);

  # Run the reservoir
  for i = 1:size(x, 1)
    update = tanh(W_in * [1; x(i, :)'] + W * internal_state);
    internal_state = (1 - leaking_rate) * internal_state + leaking_rate * update;

    if (i > flush_length)
      final_internal_state(i - flush_length, :) = [1, x(i, :), internal_state'];
    end
  end

  targets = x(flush_length+2:end, 3);
  data = final_internal_state(1:end - 1, :);

  W_out = pinv(data' * data) * data' * targets;

  predicted = data * W_out;
end