function y = activate(x)
	y = ((1 ./ (1+exp(-x))) .* 2) - 1;
endfunction
