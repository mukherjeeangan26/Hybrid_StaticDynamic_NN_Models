function [ynn_t,nn_dyn,Xi,Ai] = TrainNLD(Imat_t,dsr_t,nh,no,tn)

X = tonndata(Imat_t,true,false);
T = tonndata(dsr_t,true,false);

trainFcn = 'trainscg';  

inputDelays = 0;
feedbackDelays = 1;
hiddenLayerSize = nh;
netnarx_d = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'closed',trainFcn);

netnarx_d.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
netnarx_d.trainParam.epochs = 1000;
netnarx_d.trainParam.max_fail = 500;
netnarx_d.layers{1}.transferFcn = 'logsig';
netnarx_d.layers{2}.transferFcn = 'tansig';

[x,xi,ai,t] = preparets(netnarx_d,X,{},T);

netnarx_d.performFcn = 'msereg';  

[netnarx_d,tr_d] = train(netnarx_d,x,t,xi,ai);

nn_dyn = netnarx_d;
Xi = xi; Ai = ai;

y = netnarx_d(x,Xi,Ai);
    
yn = zeros(no,tn-1);
for i = 1:tn-1
    yn(1:no,i) = y{i};
end
    
ynn_t(1,:) = dsr_t(:,1);
      
for i = 1:no
    ynn_t(2:tn,i) = (yn(i,:))';
end

end