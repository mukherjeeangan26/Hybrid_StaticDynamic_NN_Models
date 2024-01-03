function ynn_v = ValNLDNLS(Imat_v,dsr_init,nn_stat,nn_dyn,Xi,Ai,tv,no,nh)

X = tonndata(Imat_v,true,false);

yv = nn_dyn(X,Xi,Ai);

yn = zeros(nh,tv-1);
for i = 1:tv-1
    yn(1:nh,i) = yv{i};
end

ynn_d = zeros(tv,nh);

ynn_d(1,1:nh) = 0;

for i = 1:nh
    ynn_d(2:tv,i) = abs((yn(i,:))');
end

ynn_v(1,1:no) = dsr_init;

ynn_v(2:tv,:) = (sim(nn_stat,(ynn_d(2:tv,:))'))';

end