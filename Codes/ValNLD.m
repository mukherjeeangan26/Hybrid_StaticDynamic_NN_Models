function ynn_v = ValNLD(Imat_v,dsr_init,nn_dyn,Xi,Ai,tv,no)

u_v = Imat_v;

X = tonndata(u_v,true,false);

yv = nn_dyn(X,Xi,Ai);

yn = zeros(no,tv-1);
for i = 1:tv-1
    yn(1:no,i) = yv{i};
end

ynn_v = zeros(tv,no);

ynn_v(1,1:no) = dsr_init;
      
 for i = 1:no
     ynn_v(2:tv,i) = (yn(i,:))';
 end

end