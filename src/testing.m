Intype = [];
Outtype = [];
Errortype = [];

figure(10)
plot(X,Y,'k');
hold on;

output = [];
for i=1:length(testset)
    output = [output; feedforward(testset(i,1:2),w)'];
    if output(i,1) < output(i,2)
        Intype = [Intype; testset(i,1:2)];
    else
        Outtype = [Outtype; testset(i,1:2)];
    endif
endfor


plot(Intype(:,1),Intype(:,2),'g*');
plot(Outtype(:,1),Outtype(:,2),'g*');
hold off;
