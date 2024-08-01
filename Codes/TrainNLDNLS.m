function [ynn_t,nn_stat,nn_dyn,Xi,Ai] = TrainNLDNLS(Imat_t,dsr_t,nh,no,tn)

maxiter = 10;

int_mat_1 = zeros(nh,tn);

check1 = Inf;

x_in = Imat_t;

net_s = newff(Imat_t,dsr_t,nh,{'logsig','purelin'},'trainlm');
net_s.trainParam.max_fail = 1000;

% Initializing the Termination Criteria

term_crit = Inf; tol = 1e-2;

iter = 1;

while iter <= maxiter && term_crit >= tol
    
    net_s.trainParam.epochs = 1000;
         
    [net_s,tr_s] = train(net_s,x_in,dsr_t);
    
    y = sim(net_s,x_in);
    
    ynn_t = y';
    
    % Target values from dynamic network
    
    for i = 1:nh
        int_mat_1(i,:) = x_in(i,:);
    end
    
    X = tonndata(Imat_t,true,false);
    T = tonndata(x_in,true,false);

    trainFcn = 'trainbfg';  

    inputDelays = 0;
    feedbackDelays = 1;
    hiddenLayerSize = nh;
    netnarx_d = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'closed',trainFcn);

    netnarx_d.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
    netnarx_d.trainParam.epochs = 2000;
    netnarx_d.trainParam.max_fail = 1000;
    netnarx_d.divideFcn = 'divideblock';
    netnarx_d.layers{1}.transferFcn = 'tansig';
    netnarx_d.layers{2}.transferFcn = 'purelin';

    [x,xi,ai,t] = preparets(netnarx_d,X,{},T);

    netnarx_d.performFcn = 'msereg';  

    [netnarx_d,tr_d] = train(netnarx_d,x,t,xi,ai);
    
    y = netnarx_d(x,xi,ai);
    
    yn = zeros(nh,tn-1);
    for i = 1:tn-1
        yn(1:nh,i) = y{i};
    end
    
    ynn_d_t(1,:) = int_mat_1(:,1);
      
    for i = 1:nh
        ynn_d_t(2:tn,i) = (yn(i,:))';
    end
    
    % Updating intermediate inputs by Direct Substitution
    
    for i = 1:nh
        x_in(i,:) = ynn_d_t(:,i);
    end
    
    sse = sum((int_mat_1 - ynn_d_t').^2);
    mse = (1/(nh*tn))*sum(sse,'all');
    
    if mse <= check1        
        nn_stat = net_s;
        nn_dyn = netnarx_d;
        Xi = xi;
        Ai = ai;
        ynn_final = ynn_t;
        check1 = mse;
    end
    
    x_in_1 = x_in;
    
    if iter > 1
        term_crit = norm(x_in_1 - x_in);
    end
    
    iter = iter + 1;
    
end

ynn_t = ynn_final;

end
