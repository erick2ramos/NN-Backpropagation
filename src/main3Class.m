maxEpochs = 3000;
n_hidden = 10;
learning_rate = 0.01;
percentage = 0.7;

inp = dlmread('../datos/iris.data.modified3class.txt');
warning('off');

#  Area de aceptacion
X1 = linspace(min(inp(1:50,1)), max(inp(1:50,1)),100);
Y1 = linspace(min(inp(1:50,2)), max(inp(1:50,2)),100); 
X2 = linspace(min(inp(1:50,3)), max(inp(1:50,3)),100);
Y2 = linspace(min(inp(1:50,4)), max(inp(1:50,4)),100); 

inp = inp(randperm(size(inp,1)),:);

# Preparando datos
inputs = inp(1:floor(size(inp,1) * percentage),1:size(inp,2) - 1);
targets = inp(1:floor(size(inp,1)* percentage),size(inp,2));

validset = inp(floor(size(inp,1) * percentage):size(inp,1),:);
validin = validset(:,1:size(inp,2) - 1);
validout = validset(:,size(inp,2));

inputs = [ inputs ones(size(inputs,1),1) ];
validin = [ validin ones(size(validin,1),1) ];


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
plot(err, '.');

epocherror = 0;
figure(2);
subplot(1,2,1);
plot(X1,Y1,'k');
hold on;
subplot(1,2,2);
plot(X2,Y2,'k');
hold on;
for i = 1 : length(predictions);
    if round(predictions(i)) == 0
        if validset(i, size(validset,2)) == 0;
            subplot(1,2,1);
            plot(validset(i,1), validset(i,2), 'g+');
            subplot(1,2,2);
            plot(validset(i,3), validset(i,4), 'g+');
        else
            epocherror += 1;
            subplot(1,2,1);
            plot(validset(i,1), validset(i,2), 'ro');
            subplot(1,2,2);
            plot(validset(i,3), validset(i,4), 'ro');
        endif
    elseif round(predictions(i)) == 1
        if validset(i,size(validset,2)) == 1;
            subplot(1,2,1);
            plot(validset(i,1),validset(i,2), 'b+');
            subplot(1,2,2);
            plot(validset(i,3),validset(i,4), 'b+');
        else
            epocherror += 1;
            subplot(1,2,1);
            plot(validset(i,1), validset(i,2), 'ro');
            subplot(1,2,2);
            plot(validset(i,3), validset(i,4), 'ro');
        endif
    elseif round(predictions(i)) == 2
        if validset(i,size(validset,2)) == 2;
            subplot(1,2,1);
            plot(validset(i,1),validset(i,2), 'm+');
            subplot(1,2,2);
            plot(validset(i,3),validset(i,4), 'm+');
        else
            epocherror += 1;
            subplot(1,2,1);
            plot(validset(i,1), validset(i,2), 'ro');
            subplot(1,2,2);
            plot(validset(i,3), validset(i,4), 'ro');
        endif
    endif
endfor
fprintf("Errores: %d (%d)\nEfectividad: %f\n", epocherror, length(validset),1 - epocherror/length(validset))
hold off;
