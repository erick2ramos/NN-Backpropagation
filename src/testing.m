Intype = [];
Outtype = [];
Errortype = [];

figure(10)
plot(X,Y,'k');
hold on;

sol_mean = mean(testset(:,3));
sol_sd = std(testset(:,3));

output = [];
for i=1:length(testset)
    output = [output; feedforward(testset(i,1:2),w)'];
    if output(i) < 0.5
        Intype = [Intype; (testset(i,1:2) .* std_dev_tt(1:2)) + mu_tt(1:2)];
    else
        Outtype = [Outtype; (testset(i,1:2) .* std_dev_tt(1:2)) + mu_tt(1:2)];
    endif
endfor

if length(Intype) > 0
    plot(Intype(:,1),Intype(:,2),'g*');
endif
if length(Outtype) > 0
    plot(Outtype(:,1),Outtype(:,2),'r*');
endif
if length(Errortype) > 0
    plot(Errortype(:,1),Errortype(:,2),'d*');
endif
hold off;
