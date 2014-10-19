function input = getinput(n)
    global inp ;
    input = zeros(5, 2) ;
    for k=1:2
        a = inp(n, k) ;
        b = (k * 10) - 9 ;
        input(b:(b + a - 1)) = ones(a, 2) ;
    endfor
endfunction
