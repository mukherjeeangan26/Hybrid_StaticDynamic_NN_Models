clc
clear

% This code requires the MATLAB Neural Network and Deep Learning Toolbox
% Packages.

% This code develops and compares the different neural network (NN)
% architectures considered for modeling any generic nonlinear dynamic data.
% The candidate network models considered in this code for comparison are
% 1. Nonlinear Static (NLS) Network (Feedforward 3-layered NN)
% 2. Nonlinear Dynamic (NLD) Network (NARX-type Recurrent NN)
% 3. Hybrid Series (NLS - NLD) Static-Dynamic NN
% 4. Hybrid Series (NLD - NLS) Dynamic-Static NN
% 5. Hybrid Parallel (NLS || NLD) Static-Dynamic NN

% For more details about the structures and architectures of the hybrid all
% nonlinear series and parallel network models, refer to the corresponding
% publication: Mukherjee, A. & Bhattacharyya, D. Hybrid Series/Parallel 
% All-Nonlinear Dynamic-Static Neural Networks: Development, Training, and 
% Application to Chemical Processes. Ind. Eng. Chem. Res. 62, 3221–3237 (2023). 
% Available online at: pubs.acs.org/doi/full/10.1021/acs.iecr.2c03339 

% Load the training and validation datasets and specify the input and
% output variables to the NN models
% Note that the user can consider any dynamic dataset for training and
% validation. The rows signify the time steps and the columns signify the 
% input and output variables.

load('input_data_1.mat'); 
load('output_data_1.mat');

input_data = input_data_1; output_data = output_data_1;

ni = size(input_data,2);             % Number of inputs = Number of neurons in input layer
no = size(output_data,2);            % Number of outputs = Number of neurons in output layer

nh = ni;

nt = ni + no;

% Normalization of inputs and outputs for data preparation

data = [input_data, output_data];
tt = size(data,1);
tn = floor(0.7*tt);                % Selecting 70% of total data for training

norm_mat = zeros(tt,nt);
delta = zeros(1,nt);
for i = 1:nt
    delta(1,i) = (max(data(:,i)) - min(data(:,i)));
    norm_mat(:,i) = (data(:,i)-min(data(:,i)))/(delta(1,i));
end

Imat = (norm_mat(:,1:ni))';
dsr = (norm_mat(:,ni+1:ni+no))';

disp('The candidate NN models that can be developed in this code are: ')
disp('1. Nonlinear Static (NLS) NN')
disp('2. Nonlinear Dynamic (NLD) NN')
disp('3. Hybrid Series (NLS - NLD) Static-Dynamic NN')
disp('4. Hybrid Series (NLD - NLS) Dynamic-Static NN')
disp('5. Hybrid Parallel (NLS || NLD) Static-Dynamic NN')

network_type = input("Enter the index of NN to be developed from the above list: ");

%-----------------------------------------------------------------------%

disp('TRAINING STARTS')

tr_steps = (1:tn)';

dsr_t = zeros(no,tn);
Imat_t = zeros(ni,tn);

for i = 1:tn    
    ts = tr_steps(i,1);    
    dsr_t(1:no,i) = dsr(1:no,ts);
    Imat_t(1:ni,i) = Imat(1:ni,ts);    
end

switch network_type
    case 1
        [ynn_t,nn_stat] = TrainNLS(Imat_t,dsr_t,nh,no);
    case 2
        [ynn_t,nn_dyn,Xi,Ai] = TrainNLD(Imat_t,dsr_t,nh,no,tn);
    case 3
        [ynn_t,nn_stat,nn_dyn,Xi,Ai] = TrainNLSNLD(Imat_t,dsr_t,nh,no,tn);
    case 4
        [ynn_t,nn_stat,nn_dyn,Xi,Ai] = TrainNLDNLS(Imat_t,dsr_t,nh,no,tn);
    case 5
        [ynn_t,nn_stat,nn_dyn,Xi,Ai] = TrainNLSprlNLD(Imat_t,dsr_t,nh,no,tn);
end

disp('TRAINING ENDS')

%-----------------------------------------------------------------------%

disp('VALIDATION STARTS')

flag = 1;
tv = tt - tn;
val_steps = zeros(tv,1);

for i = 1:tt
    check = ismember(i,tr_steps);    
    if check == 0
        val_steps(flag,1) = i;
        flag = flag+1;
    end
end

val_steps = sort(val_steps);

dsr_v = zeros(no,tv);
Imat_v = zeros(ni,tv);

for i = 1:tv    
    ts = val_steps(i,1);    
    dsr_v(1:no,i) = dsr(1:no,ts);
    Imat_v(1:ni,i) = Imat(1:ni,ts);    
end

% Exposing both training and validation datasets to the optimal model

Imat_v = Imat; dsr_v = dsr;
val_steps = 1:size(Imat_v,2);
tv = tt;

dsr_init = dsr_v(:,1);

switch network_type
    case 1
        ynn_v = ValNLS(Imat_v,nn_stat);
    case 2
        ynn_v = ValNLD(Imat_v,dsr_init,nn_dyn,Xi,Ai,tv,no);
    case 3
        ynn_v = ValNLSNLD(Imat_v,dsr_init,nn_stat,nn_dyn,Xi,Ai,tv,no);
    case 4
        ynn_v = ValNLDNLS(Imat_v,dsr_init,nn_stat,nn_dyn,Xi,Ai,tv,no,nh);
    case 5
        ynn_v = ValNLSprlNLD(Imat_v,dsr_init,nn_stat,nn_dyn,Xi,Ai,tv,no);
end

disp('VALIDATION ENDS')

%-----------------------------------------------------------------------%

% Conversion of normalized scale to absolute scale

dsr_t_p = zeros(tn,no); ynn_t_p = zeros(tn,no);
dsr_v_p = zeros(tv,no); ynn_v_p = zeros(tv,no);

for i = 1:no    
    dsr_t_p(:,i) = (dsr_t(i,:))'.*delta(1,ni+i) + min(data(:,ni+i));
    dsr_v_p(:,i) = (dsr_v(i,:))'.*delta(1,ni+i) + min(data(:,ni+i));
    
    ynn_t_p(:,i) = ynn_t(:,i).*delta(1,ni+i) + min(data(:,ni+i));
    ynn_v_p(:,i) = ynn_v(:,i).*delta(1,ni+i) + min(data(:,ni+i));        
end

% Training and Validation Plots (assuming all data exposed to the models
% during validation). The black dotted line shows partition between
% training and validation datasets.

% For separate training and validation plots, the corresponding output
% variables may be plotted separately

for i = 1:no
    
    figure(i)
    hold on
    plot(dsr_v_p(:,i),'b','LineWidth',1.5)
    plot(ynn_v_p(:,i),'r--','LineWidth',1.2)
    xline(tn,'k-.','LineWidth',2.0)
    xlabel('Time (mins)')
    ylabel({'Output';i})
    switch network_type
        case 1
            title('Results from NLS NN Model')
        case 2
            title('Results from NLD NN Model')
        case 3
            title('Results from NLS-NLD NN Model')
        case 4
            title('Results from NLD-NLS NN Model')
        case 5
            title('Results from NLS || NLD NN Model')
    end
    legend('Measurements','NN','Location','northeast')
    grid on
    a=findobj(gcf);
    allaxes=findall(a,'Type','axes'); alltext=findall(a,'Type','text');
    set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14)
    set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14)
    
end


























