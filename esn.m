function [final_internal_state, W_out, W_in, W, predicted] = esn(x)

#{
To run:

x = load("stock_data_clean/KAZ_LDX.TXT" )(1:end-200,:);
x_val = load("stock_data_clean/KAZ_LDX.TXT" )(end-199:end,:);
[final_internal_state, W_out, W_in, W, predicted] = esn(x);
get_directional_accuracy(x(51:end, 3), predicted)
[final_internal_state_val, predicted_val] = esn_generate(x_val(1:end - 1, :), final_internal_state, W_out, W_in, W);
get_directional_accuracy(x_val, predicted_val)
#}

  reservoir_size = 1000
  leaking_rate = 1
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