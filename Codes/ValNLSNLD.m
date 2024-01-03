function ynn_v = ValNLSNLD(Imat_v,dsr_init,nn_stat,nn_dyn,Xi,Ai,tv,no)

u = Imat_v;
y_s = sim(nn_stat,u);

X = tonndata(y_s,true,false);

yv = nn_dyn(X,Xi,Ai);

yn = zeros(no,tv-1);
for i = 1:tv-1
    yn(1:no,i) = yv{i};
end

ynn_v = zeros(tv,no);

ynn_v(1,1:no) = dsr_init;

for i = 1:no
    ynn_v(2:tv,i) = abs((yn(i,:))');
end

end