function ynn_v = ValNLS(Imat_v,nn_stat)

ynn_v = sim(nn_stat,Imat_v);
ynn_v = ynn_v';

end