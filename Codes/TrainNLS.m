function [ynn_t,nn_stat] = TrainNLS(Imat_t,dsr_t,nh,no)

u = Imat_t;

net_s = feedforwardnet(nh ,'trainlm');

net_s.layers{1}.transferFcn = 'logsig';
net_s.layers{2}.transferFcn = 'logsig';
net_s.trainParam.max_fail = 2000;

net_s = init(net_s);
net_s.trainParam.epochs = 5000;

[net_s,tr_s] = train(net_s,u,dsr_t);

nn_stat = net_s;
ynn_t = (sim(nn_stat,u))';

end