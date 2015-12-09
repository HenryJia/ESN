function test()
spectral_radius = 0.8;
density = 0.95;
reservoir_size = 100;
in_dim = 5 + 4 * 4 + 4 + 1;
#in_dim = 5 + 1;
W_in_radius = sqrt(6 / (reservoir_size + in_dim));
#W_in_radius = 0.5;
#W_in_radius = 1 / sqrt(in_dim)
[W_in, W] = gen_reservoir(reservoir_size, in_dim, W_in_radius, density, spectral_radius);

leaking_rate = 1;
x = load("stock_data_clean/RDSA_LDX.TXT" )(1:end-200,:);
x_val = load("stock_data_clean/RDSA_LDX.TXT" )(end-199:end,:);
ftse = load("FTSE.TXT" )(end-(200-1)-size(x, 1):end-200,1:4);
ftse_val = load("FTSE.TXT" )(end-199:end,1:4);
x = [x, movavg(x, 5, 5, 0)(:, 1:4), movavg(x, 10, 10, 0)(:, 1:4), movavg(x, 15, 15, 0)(:, 1:4), movavg(x, 20, 20, 0)(:, 1:4), ftse];
x_val = [x_val, movavg(x_val, 5, 5, 0)(:, 1:4), movavg(x_val, 10, 10, 0)(:, 1:4), movavg(x_val, 15, 15, 0)(:, 1:4), movavg(x_val, 20, 20, 0)(:, 1:4), ftse_val];
[x_norm, mu, sigma] = featureNormalize(x);
x_val_norm = featureNormalizeKnown(x_val, mu, sigma);
[final_internal_state, W_out,  predicted] = esn(x_norm, W_in, W, leaking_rate);
get_directional_accuracy(x_norm(51:end, 4), predicted)
[final_internal_state_val, predicted_val] = esn_generate(x_val_norm(1:end - 1, :), final_internal_state, W_out, W_in, W, leaking_rate);
get_directional_accuracy(x_val_norm, predicted_val)
mean(abs(featureRecover(predicted, mu, sigma) ./ x(52:end, 4) - 1))
mean(abs(featureRecover(predicted_val, mu, sigma) ./ x_val(2:end, 4) - 1))
figure(1)
hold off
plot(1:size(x_val, 1)-1, x_val_norm(2:end, 4), "b", "linewidth", 1)
hold on
plot(1:size(x_val, 1)-1, predicted_val, "r", "linewidth", 1)
hold off

end