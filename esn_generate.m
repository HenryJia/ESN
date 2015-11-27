function [final_internal_state, predicted] = esn_generate(x, initial_internal_state, W_out, W_in, W, leaking_rate = 1)
  in_dim = size(x, 2) + 1;

  internal_state = initial_internal_state';
  final_internal_state = zeros(size(x, 1), size(W, 1) + in_dim);

  # Run the reservoir
  for i = 1:size(x, 1)
    update = tanh(W_in * [1; x(i, :)'] + W * internal_state);
    internal_state = (1 - leaking_rate) * internal_state + leaking_rate * update;
    final_internal_state(i, :) = [1, x(i, :), internal_state'];
  end

  predicted = final_internal_state * W_out;
  final_internal_state = final_internal_state(end, :);
end