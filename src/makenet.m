function [activations, weights] = makenet(sl)
	L = length(sl);
	weights = cell(L,1);
	activations = cell(L,1);
	activations{1} = [ 1; zeros(sl(1), 1) ];
	weights{1} = [ ones(sl(2),1) initweights(sl(2),sl(1)) ];
	for ii = 2:(L - 1)
		activations{ii} = [ 1; zeros(sl(ii), 1) ];
		weights{ii} = [ ones(sl(ii + 1), 1) initweights(sl(ii + 1), sl(ii)) ];
	endfor
	activations(L) = zeros(sl(L), 1);
endfunction

