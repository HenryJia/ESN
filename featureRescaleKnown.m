function x_res = featureRescaleKnown(x, x_max, x_min)
  x_res = (x - x_min) / (x_max - x_min);
end