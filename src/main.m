clear all

global inp ;
inp = dlmread("../datos/datos_P1_2_SD2014_n500.txt") ;

[a, w] = makenet([2, 8, 2]) ;
#load("weights.oct","w");
maxiter = 1000 ;
itr = 0 ;
trained = false ;
epsilon = 0.01 ;

checkf = 50 ;
mnerror = -1 ;
maxerrors = zeros(maxiter, 1) ;
nerrors = zeros(maxiter, 1) ;

trainingset = inp(1:floor(length(inp)*0.7),:);
#trainingset = makepoints(5000);
validset = inp(floor(length(inp)*0.7) + 1:length(inp),:);
#validset = makepoints(1000);
testset = makepoints(floor(length(inp)*0.6));
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
  TrPerm = trainingset(randperm(length(trainingset)),:);
  for n=1:length(TrPerm)
    invec = TrPerm(n,1:2) ;
    target = TrPerm(n,3) ;
    a = forwardprop(invec, a, w) ;
    if target == -1
        a{length(a)}(1) = 1 - a{length(a)}(1);
        a{length(a)}(2) = 0 - a{length(a)}(2);
    else
        a{length(a)}(1) = 0 - a{length(a)}(1);
        a{length(a)}(2) = 1 - a{length(a)}(2);
    endif
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
    target = validset(n,3) ;
    output = feedforward(invec, w);
    if target == -1
        output(1) = 1 - output(1);
        output(2) = 0 - output(2);
    else
        output(1) = 0 - output(1);
        output(2) = 1 - output(2);
    endif
    OutputPerEpoch = [OutputPerEpoch; output(1) output(2)];
    if output(1) < output(2)
        OutputErrOut = [OutputErrOut; output(1)];
        if target == -1    
            Etype = [Etype; validset(n,1:2)];
        else
            Atype = [Atype; validset(n,1:2)];
        endif
    else
        OutputErrIn = [OutputErrIn; output(2)];
        if target == 1
            Etype = [Etype; validset(n,1:2)];
        else
            Btype = [Btype; validset(n,1:2)];
        endif
    endif
    
    bias = [1; 0];
    if target == -1
        bias = [0; 1];
    endif
    
    newerror = max((target - (output - bias)).^2) ;
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
  elseif checkitr && (nerrors(itr) == mnerror)
    trained = true ;
    disp(["Training completed, the error rate has stabilized to " num2str(mnerror) " errors per validation."])
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
plot(OutputErrOut,'ro');
hold on;
plot(OutputErrIn,'g+');
hold off;

figure(5);
plot(OutputPerEpoch(:,1),"b*");
hold on;
plot(OutputPerEpoch(:,2),"g*");
hold off;

save('-binary', 'weights.oct', 'w')
