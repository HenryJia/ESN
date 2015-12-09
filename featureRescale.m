function [x_res, x_max, x_min] = featureRescale(x)
  x_max = max(x);
  x_min = min(x);
  x_res = (x - x_min) / (x_max - x_min);
end