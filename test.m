function W_out = test()

rand("seed", 1e9 * now());
#rand("seed", 0);

spectral_radius = 0.6; # Default: 0.8
density = 0.1 * 2^(-3); # Default: 0.95
reservoir_size = 800; # Default: 100
in_dim = 12 + 1;
W_in_radius = sqrt(6 / (reservoir_size + in_dim));
#W_in_radius = 0.1;
#W_in_radius = 1 / sqrt(in_dim)
flush_length = 50;
[W_in, W] = gen_reservoir(reservoir_size, in_dim, W_in_radius, density, spectral_radius);
W_b = rescale(rand(reservoir_size, 1));

leaking_rate = 1;
data = load("stock_data_clean/RDSA_LDX.TXT" )(1:end-200,:);
data_val = load("stock_data_clean/RDSA_LDX.TXT" )(end-199:end,:);
[x, y] = gen_data(data);
[x_val, y_val] = gen_data(data_val);
[x_norm, mu, sigma] = featureNormalize(x);
x_val_norm = featureNormalizeKnown(x_val, mu, sigma);
fflush(stdout);

[final_internal_state, W_out,  predicted] = esn(x_norm, y, W_in, W, W_b, leaking_rate, flush_length);
[final_internal_state_val, predicted_val] = esn_generate(x_val_norm, final_internal_state, W_out, W_in, W, W_b, leaking_rate);
#get_directional_accuracy(x(51:end, 6), predicted(1:end-1))
#get_directional_accuracy(x_val(:, 6), predicted_val(1:end-1))
mean(abs(y_val(1:end-1) ./ y_val(2:end) - 1))
mean(abs(predicted(1:end) ./ y(flush_length + 1:end) - 1))
mean(abs(predicted_val(1 + 19:end) ./ y_val(1 + 19:end) - 1))
figure(1)
hold off
plot(1:size(x_val, 1) - 19, y_val(1 + 19:end), "b", "linewidth", 1)
hold on
plot(1:size(x_val, 1) - 19, predicted_val(1 + 19:end), "r", "linewidth", 1)
hold off
figure(2)
hold off
plot(1:200, y(end-199:end), "b", "linewidth", 1)
hold on
plot(1:200, predicted(end-199:end), "r", "linewidth", 1)
hold off

#mean((predicted_val(1 + 19:end) - y_val(1 + 19:end)) .^ 2) / mean((y_val(1 + 19:end) - mean(y_val(1 + 19:end))) .^ 2)
1 - mean((predicted(1:end) - y(flush_length + 1:end)) .^ 2) / var(y(flush_length + 1:end))
1 - mean((predicted_val(1 + 19:end) - y_val(1 + 19:end)) .^ 2) / var(y_val(1 + 19:end))

mean(W_out(2:13))
mean(W_out(14:end))

end