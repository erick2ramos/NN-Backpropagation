clear all

global inp ;
inp = dlmread("../datos/datos_P1_2_SD2014_n500.txt") ;

[a, w] = makenet([2, 8, 2]) ;

maxiter = 1000 ;
itr = 0 ;
trained = false ;
epsilon = 0.01 ;

checkf = 50 ;
mnerror = -1 ;
maxerrors = zeros(maxiter, 1) ;
nerrors = zeros(maxiter, 1) ;

while ! trained
    itr += 1;
    for pattern=1:length(inp)
        for l=1:length(a)
            a = forwardprop(inp(pattern,1:2), a, w);
            a{length(a)} = inp(pattern,3) - a{length()}
            w = backprop(inp(pattern,3), a, w);
        endfor
    endfor
endwhile
w
