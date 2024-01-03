function ynn_v = ValNLSprlNLD(Imat_v,dsr_init,nn_stat,nn_dyn,Xi,Ai,tv,no)

u_v = Imat_v;

y_s_v = sim(nn_stat,u_v);

y_d_init = dsr_init - y_s_v(:,1);

X = tonndata(u_v,true,false);
yv = nn_dyn(X,Xi,Ai);

yn = zeros(no,tv-1);
for i = 1:tv-1
    yn(1:no,i) = yv{i};
end

ynn_v = zeros(tv,no);

ynn_v(1,1:no) = y_d_init;

for i = 1:no
    ynn_v(2:tv,i) = abs((yn(i,:))');
end

y_final_v = y_s_v' + ynn_v;

ynn_v = y_final_v;

end