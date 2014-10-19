function target = gettarget(n)
    global inp ;
    target = zeros(2, 1) ;
    a = inp(n, 10) + 1 ;
    target(a) = 1 ;
endfunction
