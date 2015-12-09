function p = percentageChange(x)
  data = x;
  data(:, end) += 1;
  p = data(2:end, :) ./ data(1:end - 1, :) - 1;
end