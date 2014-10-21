function [ w_i_h, w_h_o ] = backpropagation(lr, input, activations, errors, wih, who)
    d_h_o = errors .* (lr/10) .* activations';
    w_h_o = who - d_h_o';
    d_i_h = lr .* errors .* w_h_o' .* (1 - (activations').^2 ) .* input;
    w_i_h = wih - d_i_h';
endfunction
