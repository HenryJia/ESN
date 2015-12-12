function [W_in, W] = gen_reservoir(reservoir_size, in_dim, W_in_radius = 1, density = 0.01, spectral_radius = 1)

  W_in = rescale(rand(reservoir_size, in_dim), W_in_radius);

  W = sprand(reservoir_size, reservoir_size, density);
  W = spfun(@rescale, W);

  sr = abs(eigs(W, 1))
  W = W / sr * spectral_radius;
  sr = abs(eigs(W, 1))
end