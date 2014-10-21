function [ activation , errors ] = forwardpropagation(input, target, w_i_h, w_h_o)
    activation = activate(input * w_i_h); 
    output = activation * w_h_o';
    errors = output - target;
endfunction
