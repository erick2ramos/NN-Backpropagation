maxEpochs = 3000;
n_hidden = 10;
learning_rate = 0.01;

#inp = dlmread('../datos/datos_P1_2_SD2014_n500.txt');
[ inp, tnp ] = makepoints(500);
warning('off');

# Preparando datos
inputs = inp(:,1:2);
targets = inp(:,3);

grid = linspace(0, 20, 40)';
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
        validset((i-1)*length(grid)+j,:) = [ x_g_in y_g_in g_in ];
    endfor
endfor
validin = validset(:,1:2);
validout = validset(:,3);

inputs = [ inputs ones(length(inputs),1) ];
validin = [ validin ones(length(validin),1) ];

# Vector del circulo para graficar
angle = 0;
increment = 2*pi/(100);
for i = 1:101
    X(i,:) = 7*cos(angle) + 10;
    Y(i,:) = 7*sin(angle) + 10;
    angle += increment;
end

# Iniciando los pesos en valores aleatorios [-0.5, 0.5]
w_i_h = initweights(size(inputs,2),n_hidden);
w_h_o = initweights(1,n_hidden);

tic;
for epoch = 1:maxEpochs
    #Entrenamiento
    for i = 1:size(inputs,1)
        random_input = round(rand * size(inputs,1) + 0.5);
        t_input = inputs(i,:);
        t_target = targets(i,1);

        [ activations, errors ] = forwardpropagation(t_input, t_target, w_i_h, w_h_o);
        [ w_i_h, w_h_o ] = backpropagation(learning_rate, t_input, activations, errors, w_i_h, w_h_o);
    endfor

    #Validacion con la grilla de puntos
    predictions = w_h_o * activate(validin * w_i_h)';
    error = predictions' - validout;
    err(epoch) = (sum(error.^2))^0.5;
    if err(epoch) <  0.01;
        fprintf("Sali porque aprendi\n")
        break
    endif
endfor
toc

# Graficando el error cuadratico medio
figure(1);
plot(err, '*');

epoch_errors = 0;
figure(2);
plot(X,Y,'k');
hold on;
for i = 1 : length(predictions);
    if predictions(i) < 0
        if validset(i, 3) == -1;
            plot(validset(i,1), validset(i,2), 'g*');
        else
            epoch_errors += 1;
            plot(validset(i,1), validset(i,2), 'r*');
        endif
    else
        if validset(i,3) == 1;
            plot(validset(i,1),validset(i,2), 'b+');
        else
            epoch_errors += 1;
            plot(validset(i,1), validset(i,2), 'm*');
        endif
    endif
endfor
hold off;

fprintf("Errores: %d (%d)\nEfectividad: %f\n", epoch_errors, length(validset),1 - epoch_errors/length(validset))
