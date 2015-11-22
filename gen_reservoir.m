function [W_in, W] = gen_reservoir(reservoir_size, in_dim, W_in_radius = 1, W_radius, density = 0.01, spectral_radius = 1)
  rand('seed', 0);

  W_in = rescale(rand(reservoir_size, in_dim), W_in_radius);

  W = sprand(reservoir_size, reservoir_size, density);
  W = spfun(@rescale, W);

  sr = eigs(W, 1, isreal = true)

  W = W / sr * spectral_radius;
end