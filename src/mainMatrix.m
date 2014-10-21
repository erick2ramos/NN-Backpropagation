maxEpochs = 3000;
n_hidden = 2;
learning_rate = 0.01;

inp = dlmread('../datos/datos_P1_2_SD2014_n500.txt');
warning('off');

# Normalizando datos
inputs = inp(:,1:2);
targets = inp(:,3);

inputs_mean = mean(inputs);
inputs_std = std(inputs);
inputs = (inputs(:,:) - inputs_mean(:,1)) / inputs_std(:,1);

targets = targets';
targets_mean = mean(targets);
targets_std = std(targets);
targets = (targets(:,:) - targets_mean(:,1)) / targets_std(:,1);
targets = targets';

[ validset, testset ] = makepoints(100);
grid = linspace(0, 20, 100)';
validset = zeros(length(grid)^2,3);
for i = 1:length(grid)
    x_g_in = grid(i);
    for j = 1:length(grid)
        y_g_in = grid(j);
        if sqrt((x_g_in-10)^2 + (y_g_in-10)^2) < 7
            g_in = -1;
        else
            g_in = 1;
        endif
        validset(i*length(grid)+j,:) = [ x_g_in y_g_in g_in ];
    endfor
endfor
validin = validset(:,1:2);
validout = validset(:,3);
validin_mean = mean(validin);
validin_std = std(validin);
validin = (validin(:,:) - validin_mean(:,1)) / validin_std(:,1);
validout_mean = mean(validout);
validout_std = std(validout);
validout = (validout(:,:) - validout_mean(:,1)) / validout_std(:,1);

inputs = [ inputs ones(length(inputs),1) ];
validin = [ validin ones(length(validin),1) ];

# Vector del circulo
angle = 0;
increment = 2*pi/(100);
for i = 1:101
    X(i,:) = 7*cos(angle) + 10;
    Y(i,:) = 7*sin(angle) + 10;
    angle += increment;
end
figure(1);

w_i_h = initweights(size(inputs,2),n_hidden);
w_h_o = initweights(1,n_hidden);

tic;
for epoch = 1:maxEpochs
    #Entrenamiento
    for i = 1:size(inputs,1)
        random_input = round(rand * size(inputs,1) + 0.5);
        t_input = inputs(random_input,:);
        t_target = targets(random_input,1);

        [ activations, errors ] = forwardpropagation(t_input, t_target, w_i_h, w_h_o);
        [ w_i_h, w_h_o ] = backpropagation(learning_rate, t_input, activations, errors, w_i_h, w_h_o);
    endfor
    epoch_errors = 0;
    #Validacion
    predictions = w_h_o * activate(validin * w_i_h)';
    error = predictions' - validout;
    err(epoch) = (sum(error.^2))^0.5;
    #plot(err, '*');
    #drawnow;
    if err(epoch) < 0.01;
        fprintf("Sali porque aprendi\n");
        break
    endif
endfor
toc

figure(2);
plot(X,Y,'k');
hold on;
for i = 1 : length(predictions);
    if predictions(i) < 0
        predictions(i) = -1;
    else
        predictions(i) = 1;
    endif
    if predictions(i) != validset(i, 3);
        epoch_errors += 1;
        plot(validset(i,1), validset(i,2), 'r*');
        #drawnow;
    elseif predictions(i) < 0
        plot(validset(i,1), validset(i,2), 'g*');
        #drawnow;
    else
        plot(validset(i,1), validset(i,2), 'b*');
        #drawnow;
    endif
endfor
hold off;

