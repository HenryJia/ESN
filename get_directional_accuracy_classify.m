function f1 = get_directional_accuracy_classify(data, predicted)
  size(data)
  size(predicted)
  correct_direction = zeros(size(predicted, 1), 1);
  predicted_direction = zeros(size(predicted, 1), 1);
  correct = zeros(size(predicted, 1), 1);

  for i = 1:size(predicted, 1)
    if (predicted(i) >= 0.5)
      predicted_direction(i) = 1;
    end
    if(data(i + 1) > data(i))
      correct_direction(i) = 1;
    end
  end

  accuracy = sum(predicted_direction == correct_direction) / size(predicted, 1)

  sum(predicted_direction)

  precision = sum(predicted_direction .* correct_direction) / sum(predicted_direction)
  recall = sum(predicted_direction .* correct_direction) / sum(correct_direction)

  f1 = 2 * (precision * recall) / (precision + recall);

end