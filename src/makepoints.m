function [Test, Train] = makepoints(nsize)
    n = nsize;
    Test = zeros(n,3);
    Test(:,1) = 20.*rand(n,1);
    Test(:,2) = 20.*rand(n,1);
    for i=1:n
        if sqrt((Test(i,1)-10)^2 + (Test(i,2)-10)^2) < 7
            Test(i,3) = 0;
        else
            Test(i,3) = 1;
        endif
    endfor

    n = nsize;
    Train = zeros(n,3);
    Train(:,1) = 20.*rand(n,1);
    Train(:,2) = 20.*rand(n,1);
    for i=1:n
        if sqrt((Train(i,1)-10)^2 + (Train(i,2)-10)^2) < 7
            Train(i,3) = 0;
        else
            Train(i,3) = 1;
        endif
    endfor
endfunction
