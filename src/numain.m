clear all

global inp ;
inp = dlmread("../datos/datos_P1_2_SD2014_n500.txt") ;
inp2 = inp(:,1:2);
m_inp = mean(inp2);
sd_inp = std(inp2);
inp2 = (inp2(:,:) - m_inp(:,1)) / sd_inp(:,1);
solutions = inp(:,3)';
m_sol = mean(solutions);
sd_sol = std(solutions);
solutions = (solutions(:,:) - m_sol(:,1)) / sd_sol(:,1);
solutions = solutions';
bias = ones(size(inp2),1);
inp2 = [inp2 bias];

[a, w] = makenet([2, 8, 1]) ;
#load("weights.oct","w");
maxiter = 1000 ;
itr = 0 ;
trained = false ;
epsilon = 0.01 ;

checkf = 50 ;
mnerror = -1 ;
maxerrors = zeros(maxiter, 1) ;
nerrors = zeros(maxiter, 1) ;

trainingset = inp2(1:floor(length(inp2)*0.7),:);
#trainingset = makepoints(5000);
validset = inp2(floor(length(inp2)*0.7) + 1:length(inp2),:);
validsol = solutions(floor(length(inp2)*0.7) + 1:length(inp2),:);
#validset = makepoints(1000);
testset = makepoints(floor(length(inp2)*0.6));
mu_tt = mean(testset);
std_dev_tt = std(testset);
testset = (testset(:,:) - mu_tt(:,1)) ./ std_dev_tt(:,1);
Atype = [];
Btype = [];
Etype = [];
angle = 0;
OutputErrIn = [];
OutputErrOut = [];
OutputPerEpoch = [];
increment = 2*pi/(100);
for i = 1:101
    X(i,:) = 7*cos(angle) + 10;
    Y(i,:) = 7*sin(angle) + 10;
    angle += increment;
end

while ! trained
  
  itr += 1 ; 
  
  # Training
  #TrPerm = trainingset(randperm(length(trainingset)),:);
  TrPerm = trainingset;
  for n=1:length(TrPerm)
    invec = TrPerm(n,1:2) ;
    target = solutions(n) ;
    a = forwardprop(invec, a, w) ;
    w = backprop(target, a, w) ;
  endfor
  
  Atype = [];
  Btype = [];
  EType = [];
  OutputErrIn = [];
  OutputErrOut = [];
  # Validation
  for n=1:length(validset)
    invec = validset(n,1:2) ;
    target = solutions(n) ;
    output = feedforward(invec, w);
    OutputPerEpoch = [OutputPerEpoch; output];
    if output < 0
        OutputErrOut = [OutputErrOut; output];
        if target == -1    
            Etype = [Etype; validset(n,1:2) .* sd_inp(1:2) + m_inp(1:2)];
        else
            Atype = [Atype; validset(n,1:2) .* sd_inp(1:2) + m_inp(1:2)];
        endif
    else
        OutputErrIn = [OutputErrIn; output];
        if target == 1
            Etype = [Etype; validset(n,1:2) .* sd_inp(1:2) + m_inp(1:2)];
        else
            Btype = [Btype; validset(n,1:2) .* sd_inp(1:2) + m_inp(1:2)];
        endif
    endif
    
    newerror = max((target - output).^2) ;
    maxerrors(itr) = max([newerror, maxerrors(itr)]) ;
    if newerror >= epsilon
      nerrors(itr) += 1 ;
    endif
  endfor 
  
  checkitr = ! mod(itr, checkf) ;
  if checkitr
    mnerror = mean(nerrors((itr-checkf+1):itr)) ;
  endif
  
  if nerrors(itr) == 0
    trained = true ;
    disp("Training completed, no errors during validation.")
#  elseif checkitr && (nerrors(itr) == mnerror)
#    trained = true ;
#    disp(["Training completed, the error rate has stabilized to " num2str(mnerror) " errors per validation."])
  elseif itr == maxiter
    trained = true ;
    disp(["Stopping due to maxiter. " num2str(maxiter)  " iterations already done."])
  endif
  
endwhile



figure(1)
plot(maxerrors(1:itr),'*')
figure(2)
plot(nerrors(1:itr),'*')

figure(3)
plot(X,Y,'k');
hold on;
if length(Atype) > 0
    plot(Atype(:,1),Atype(:,2),'r*');
endif
if length(Btype) > 0
    plot(Btype(:,1),Btype(:,2),'g*');
endif
if length(Etype) > 0
    plot(Etype(:,1),Etype(:,2),'b*');
endif
legend({'','Out','In','Error'},'location','east');
hold off;

figure(4);
if length(OutputErrOut) > 0
    plot(OutputErrOut,'ro');
    hold on;
endif
if length(OutputErrIn) > 0
    plot(OutputErrIn,'g+');
endif
hold off;

figure(5);
for i=1:size(OutputPerEpoch)(2)
    plot(OutputPerEpoch(:,i),"*");
    hold on;
endfor
hold off;

save('-binary', 'weights.oct', 'w')
