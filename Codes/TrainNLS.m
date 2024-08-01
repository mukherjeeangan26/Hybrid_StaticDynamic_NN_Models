function [ynn_t,nn_stat] = TrainNLS(Imat_t,dsr_t,nh,no)

u = Imat_t;

net_s = newff(Imat_t,dsr_t,nh,{'logsig','logsig'},'trainlm');

net_s.trainParam.max_fail = 1000;

net_s = init(net_s);
net_s.trainParam.epochs = 5000;

[net_s,tr_s] = train(net_s,u,dsr_t);

nn_stat = net_s;
ynn_t = (sim(nn_stat,u))';

end
