function [final_internal_state, W_out, predicted] = esn(x, y, W_in, W, W_b, leaking_rate = 1)

  lambda = 0;
  iters = 0;
  flush_length = 50;
  in_dim = size(x, 2) + 1;
  reservoir_size = size(W, 1);
  noise = 0;

  internal_state = rescale(rand(reservoir_size, 1));
  final_internal_state = zeros(size(x, 1) * iters - flush_length, reservoir_size + in_dim);
  update = zeros(reservoir_size, 1);

  # Run the reservoir
  for i = 1:size(x, 1)
    rand("seed", 1e9 * now());
    update = tanh(W_in * [1; x(i, :)'] + W * internal_state + W_b * y(i) / 10000 + rescale(rand(reservoir_size, 1), noise));
    #update = tanh(W_in * [1; x(i, :)'] + W * internal_state + W_b * x(i, 6));
    internal_state = (1 - leaking_rate) * internal_state + leaking_rate * update;

    if (i > flush_length)
      final_internal_state(i - flush_length, :) = [1, x(i, :), internal_state'];
    end
  end

  for j = 1:iters
  for i = 1:size(x, 1)
    rand("seed", 1e9 * now());
    update = tanh(W_in * [1; x(i, :)'] + W * internal_state + W_b * y(i) / 10000 + rescale(rand(reservoir_size, 1), noise));
    #update = tanh(W_in * [1; x(i, :)'] + W * internal_state + W_b * x(i, 6));
    internal_state = (1 - leaking_rate) * internal_state + leaking_rate * update;

    final_internal_state(j * size(x, 1) + i - flush_length, :) = [1, x(i, :), internal_state'];
  end
  end

  targets = [y(flush_length+1:end, 1); repmat(y, iters, 1)];
  data = final_internal_state;

  W_out = pinv(data' * data + lambda * eye(in_dim + reservoir_size)) * data' * targets;

  predicted = (data * W_out)(iters * size(x, 1) + 1:end);
  figure(10)
  plot(final_internal_state(end-250:end, randi([in_dim + 1, 100 + in_dim])))
  final_internal_state = final_internal_state(end, in_dim + 1:end);
end