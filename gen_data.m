function [x, y] = gen_data(data)

history = 5; # How many days further we look back not including the current price

data_needed = data(:, 4:5); # We just want the closing price and volume

ftse = load("FTSE.TXT" )(end - (200 - 1) - size(data, 1):end - 200, 4); # The FTSE 100 price

mov5 = movavg(data_needed(:, 1), 5, 5, 0);
mov10 = movavg(data_needed(:, 1), 10, 10, 0);
mov15 = movavg(data_needed(:, 1), 15, 15, 0);
mov20 = movavg(data_needed(:, 1), 20, 20, 0);

x = zeros(size(data_needed, 1) - (history + 1) - 1 + 1, 12);
y = zeros(size(data_needed, 1) - (history + 1) - 1 + 1, 1);

for i = 1:size(x, 1)
  j = 0;
  #while (j < history) # Put the historical prices into a vector
  #  x(i, j + 1) = data_needed(i + j, 1);
  #end
  x(i, 1:history) = data_needed(i:(i+history-1), 1);# Faster version
  j += history; # Then today's price
  x(i, j + 1) = data_needed(i + j, 1);
  x(i, j + 2) = data_needed(i + j, 2); # And today's volume

  x(i, j + 3) = mov5(i + j); # The 5 moving averages
  x(i, j + 4) = mov10(i + j);
  x(i, j + 5) = mov15(i + j);
  x(i, j + 6) = mov20(i + j);

  x(i, j + 7) = ftse(i + j); # Finally, add the FTSE 100

  y(i) = data_needed(i + j + 1, 1);
end

end