function [ynn_t,nn_stat,nn_dyn,Xi,Ai] = TrainNLSprlNLD(Imat_t,dsr_t,nh,no,tn)

maxiter = 10;
check1 = Inf;

u_s = Imat_t;
u_d = Imat_t;
dsr_t_s = dsr_t;

net_s = feedforwardnet(nh,'trainlm');
net_s.layers{1}.transferFcn = 'logsig';
net_s.layers{2}.transferFcn = 'purelin';
net_s.trainParam.max_fail = 1000;

% Initializing the Termination Criteria

term_crit = Inf; tol = 1e-2;

iter = 1;

while iter <= maxiter && term_crit >= tol
    
    net_s.trainParam.epochs = 2000;      
    [net_s,tr_s] = train(net_s,u_s,dsr_t_s);
    y_s = sim(net_s,u_s);
    ynn_t_s = y_s';

    y_d = dsr_t - y_s;

    X = tonndata(u_d,true,false);
    T = tonndata(y_d,true,false);

    trainFcn = 'trainbfg';  

    inputDelays = 0;
    feedbackDelays = 1;
    hiddenLayerSize = nh;
    netnarx_d = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'closed',trainFcn);

    netnarx_d.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
    netnarx_d.trainParam.epochs = 2000;      
    netnarx_d.divideFcn = 'divideblock';
    netnarx_d.trainParam.max_fail = 1000;
    netnarx_d.layers{1}.transferFcn = 'logsig';
    netnarx_d.layers{2}.transferFcn = 'tansig';

    [x,xi,ai,t] = preparets(netnarx_d,X,{},T);
    netnarx_d.performFcn = 'msereg';  

    [netnarx_d,tr_d] = train(netnarx_d,x,t,xi,ai);

    y = netnarx_d(x,xi,ai);

    yn = zeros(no,tn-1);
    for i = 1:tn-1
        yn(1:no,i) = y{i};
    end

    ynn_t_d(1,1:no) = y_d(1:no,1);

    for i = 1:no
        ynn_t_d(2:tn,i) = (yn(i,:))';
    end

    y_final = ynn_t_s + ynn_t_d;

    sse = sum((dsr_t' - y_final).^2);
    mse = (1/(no*tn))*sum(sse);
    
    if mse <= check1        
        nn_stat = net_s;
        nn_dyn = netnarx_d;
        Xi = xi;
        Ai = ai;
        ynn_final = y_final;
        check1 = mse;
    end
    
    y_d_1 = y_d;
    
    if iter > 1
        term_crit = norm(y_d_1 - y_d);
    end
    
    iter = iter + 1; 
    
end

ynn_t = ynn_final;

end