function x = featureUnscale(x_res, x_max, x_min)
  x = x_res * (x_max - x_min) + x_min;
end