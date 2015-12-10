function test()

rand("seed", now());

spectral_radius = 0.8;
density = 0.95;
reservoir_size = 100;
in_dim = 12 + 1;
W_in_radius = sqrt(6 / (reservoir_size + in_dim));
#W_in_radius = 0.5;
#W_in_radius = 1 / sqrt(in_dim)
[W_in, W] = gen_reservoir(reservoir_size, in_dim, W_in_radius, density, spectral_radius);

leaking_rate = 1;
data = load("stock_data_clean/RDSA_LDX.TXT" )(1:end-200,:);
data_val = load("stock_data_clean/RDSA_LDX.TXT" )(end-199:end,:);
[x, y] = gen_data(data);
[x_val, y_val] = gen_data(data_val);
[x_norm, mu, sigma] = featureNormalize(x);
x_val_norm = featureNormalizeKnown(x_val, mu, sigma);
[final_internal_state, W_out,  predicted] = esn(x_norm, y, W_in, W, leaking_rate);
[final_internal_state_val, predicted_val] = esn_generate(x_val_norm, final_internal_state, W_out, W_in, W, leaking_rate);
get_directional_accuracy(x(51:end, 6), predicted(1:end-1))
get_directional_accuracy(x_val(:, 6), predicted_val(1:end-1))
mean(abs(predicted(1:end - 1) ./ x(52:end, 6) - 1))
mean(abs(predicted_val(1:end - 1) ./ x_val(2:end, 6) - 1))
figure(1)
hold off
plot(1:size(x_val, 1)-1, x_val(2:end, 6), "bo", "linewidth", 1)
hold on
plot(1:size(x_val, 1)-1, predicted_val(1:end - 1), "r+", "linewidth", 1)
hold off

end