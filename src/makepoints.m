function [Train, Test] = makepoints(nsize)
    half_n = floor(nsize / 2);
    half_n_up = nsize - half_n;
    Train = zeros(nsize,3);
    in_count = 1;
    out_count = 1;
    while half_n - in_count + 1 != 0
        r_point = [ 20.*rand 20.*rand ];
        if sqrt((r_point(1)-10)^2 + (r_point(2)-10)^2) < 7
            Train(in_count, 1:2) = r_point;
            Train(in_count, 3) = -1;
            in_count += 1;
        endif
    endwhile
    in_count -= 1;
    while half_n_up - out_count + 1 != 0
        r_point = [ 20.*rand 20.*rand ];
        if sqrt((r_point(1)-10)^2 + (r_point(2)-10)^2) >= 7
            Train(out_count + in_count, 1:2) = r_point;
            Train(out_count + in_count, 3) = 1;
            out_count += 1;
        endif
    endwhile
    Train = Train(randperm(size(Train,1)),:);

    Test = zeros(nsize,3);
    Test(:,1) = 20.*rand(nsize,1);
    Test(:,2) = 20.*rand(nsize,1);
    for i=1:nsize
        if sqrt((Test(i,1)-10)^2 + (Test(i,2)-10)^2) < 7
            Test(i,3) = -1;
        else
            Test(i,3) = 1;
        endif
    endfor
endfunction
