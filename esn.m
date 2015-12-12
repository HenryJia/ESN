function [final_internal_state, W_out, predicted] = esn(x, y, W_in, W, W_b, leaking_rate = 1, flush_length = 50)

  lambda = 1e-3;
  in_dim = size(x, 2) + 1;
  reservoir_size = size(W, 1);

  internal_state = rescale(rand(reservoir_size, 1));
  final_internal_state = zeros(size(x, 1) - flush_length, reservoir_size + in_dim);
  update = zeros(reservoir_size, 1);

  # Run the reservoir
  for i = 2:size(x, 1)
    rand("seed", 1e9 * now());
    update = tanh(W_in * [1; x(i, :)'] + W * internal_state + W_b * y(i - 1) / 10000 + rescale(rand(reservoir_size, 1), 1e-1));
    #update = tanh(W_in * [1; x(i, :)'] + W * internal_state + W_b * x(i, 6) + rescale(rand(reservoir_size, 1), 5e-2));
    internal_state = (1 - leaking_rate) * internal_state + leaking_rate * update;

    if (i > flush_length)
      #final_internal_state(i - flush_length, :) = [1, x(i, :), internal_state'];
      final_internal_state(i - flush_length, :) = [1, zeros(size(x(i, :))), internal_state'];
    end
  end

  targets = y(flush_length+1:end, 1);
  data = final_internal_state;

  W_out = pinv(data' * data + lambda * eye(in_dim + reservoir_size)) * data' * targets;
  #W_out(2:in_dim) = 0;
  predicted = data * W_out;
  figure(10)
  plot(final_internal_state(end-50:end, randi([in_dim + 1, 100 + in_dim])))
  final_internal_state = final_internal_state(end, in_dim + 1:end);
end