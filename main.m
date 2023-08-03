clear; clc; close all;
%% Setup Everything

% Add the submodules to path
addpath(genpath('OFDM-Matlab'))

rms_input = 0.50;

% Setup the PA simulator or TX board
PA_board = 'webRF'; % either 'WARP', 'webRF', or 'none'

switch PA_board
    case 'WARP'
        warp_params.nBoards = 1;         % Number of boards
        warp_params.RF_port  = 'A2B';    % Broadcast from RF A to RF B. Can also do 'B2A'
        board = WARP(warp_params);
        Fs = 40e6;    % WARP board sampling rate.
    case 'none'
        board = PowerAmplifier(7, 4);
        Fs = 40e6;    % WARP board sampling rate.
    case 'webRF'
        dbm_power = -22;
        board = webRF(dbm_power);
end

% Setup OFDM
ofdm_params.nSubcarriers = 1200;
ofdm_params.subcarrier_spacing = 15e3; % 15kHz subcarrier spacing
ofdm_params.constellation = 'QPSK';
ofdm_params.cp_length = 144; % Number of samples in cyclic prefix.
ofdm_params.nSymbols = 1;
modulator = OFDM(ofdm_params);

% Create TX Data
[tx_data, ~] = modulator.use;
tx_data = Signal(tx_data, modulator.sampling_rate, rms_input);
tx_data.upsample(board.sample_rate)

% Setup DPD
dpd_params.order = 7;
dpd_params.memory_depth = 4;
dpd_params.nIterations = 3;
dpd_params.learning_rate = 0.8;
dpd_params.learning_method = 'newton'; % Or 'ema' for exponential moving average.
dpd_params.use_dc_term = 1; % Adds an additional term for DC
% DDR参数
dpd_params.model_select = 'ddr';
dpd_params.use_simplifier = 0;
% GMP参数
dpd_params.use_even = false; 
dpd_params.use_conj = 0;    % Conjugate branch. Currently only set up for MP (lag = 0)
dpd_params.lag_depth = 2;  % 0 is a standard MP. >0 is GMP.
% 产生3种DPD
dpd = ILA_DPD(dpd_params);
dpd_sp = ILA_DPD(dpd_params);
dpd_params.nIterations = 1;
dpd_omp = ILA_DPD(dpd_params);

%% Run Expierement: SP扫描
% 对照试验(1) 测试PA 
[y_pa, y_pa_orginal] = board.transmit(tx_data.data);
nmse_pa = dpd_sp.cal_nrmse(y_pa, tx_data.data, 'dB');
disp(['nmse without dpd is: ' num2str(nmse_pa) 'dB'])
acpr_pa = dpd_sp.cal_acpr(y_pa_orginal.measure_all_powers);
disp(['acpr without dpd is: ' num2str(acpr_pa) 'dB'])

% 对照试验(2) 完整dpd性能
n_coeffs_dpd = dpd.perform_learning(tx_data.data, board);
[y_dpd, y_dpd_orginal] = board.transmit(dpd.predistort(tx_data.data));
nmse_dpd = dpd.cal_nrmse(y_dpd, tx_data.data, 'dB');
acpr_dpd = dpd.cal_acpr(y_dpd_orginal.measure_all_powers);
nmse_up_dpd = nmse_dpd - nmse_pa; acpr_up_dpd = acpr_dpd - acpr_pa;

% 对照试验(3) OMP性能
spar = 25;
% n_coeffs_omp = dpd_omp.omp_learning(tx_data.data,board,spar);
% [y_omp, y_omp_orginal] = board.transmit(dpd_omp.predistort(tx_data.data));
% nmse_omp = dpd_omp.cal_nrmse(y_omp, tx_data.data, 'dB');
% acpr_omp = dpd.cal_acpr(y_omp_orginal.measure_all_powers);
% nmse_up_omp = nmse_omp - nmse_pa; acpr_up_omp = acpr_omp - acpr_pa;
for i = 1:3
    intv = 4;
    n_coeffs_omp = dpd_omp.SP_learning(tx_data.data, board, spar, intv);
    [y_omp, y_omp_orginal] = board.transmit(dpd_omp.predistort(tx_data.data));
    nmse_omp = dpd_omp.cal_nrmse(y_omp, tx_data.data, 'dB');
    acpr_omp = dpd.cal_acpr(y_omp_orginal.measure_all_powers);
    nmse_up_omp(i) = nmse_omp - nmse_pa; acpr_up_omp = acpr_omp - acpr_pa;
    dpd_omp.reset_dpd
end
nmse_up_omp = mean(nmse_up_omp);

% 测试SP性能
for i = 1:3
    syn_intv = 2;
    n_coeffs_sp = dpd_sp.SP_learning(tx_data.data, board, spar, syn_intv);
    [y_sp, y_sp_orginal] = board.transmit(dpd_sp.predistort(tx_data.data));
    nmse_sp = dpd_sp.cal_nrmse(y_sp, tx_data.data, 'dB');
    acpr_sp = dpd.cal_acpr(y_sp_orginal.measure_all_powers);
    nmse_up_sp(i) = nmse_sp - nmse_pa; acpr_up_sp = acpr_sp - acpr_pa;
    dpd_sp.reset_dpd
end
nmse_up_sp = mean(nmse_up_sp);

% 实验结果table保存
nmse_up = [nmse_up_dpd; nmse_up_sp; nmse_up_omp];
acpr_up = [acpr_up_dpd; acpr_up_sp; acpr_up_omp];
nmse_with_dpd = [nmse_dpd;nmse_sp;nmse_omp];
number_of_coeffs = [n_coeffs_dpd;n_coeffs_sp;n_coeffs_omp];

result_table = table(number_of_coeffs,nmse_up,acpr_up,nmse_with_dpd,'VariableNames', ...
    {'n_coeffs' 'nmse_up' 'acpr_up' 'nmse_with_dpd'}, 'RowName',{'dpd' 'SP_5' 'SP_0'});
disp(result_table)


%% Plot
y_pa_orginal.plot_psd;
y_dpd_orginal.plot_psd;
y_sp_orginal.plot_psd
legend('without dpd', 'with dpd', 'with sp-dpd')
% dpd.plot_history;

%% save data
nowTime = datestr(now,30);
nowTime = nowTime(5:end-2);

folderName = ['In' nowTime '_' num2str(dpd_params.order) num2str(dpd_params.memory_depth)];
switch dpd_sp.model_select
    case 'gmp'
        folderName = [folderName num2str(dpd_params.lag_depth) 'gmp_'];
    case 'ddr'
        if dpd_params.use_simplifier
            folderName = [folderName 'sddr_'];
        else
            folderName = [folderName 'ddr_'];
        end
end

folderName = [folderName dpd_params.learning_method num2str(dpd_params.nIterations)];
folderName = [folderName '_S' num2str(spar) '_intv' num2str(syn_intv)];

% folderName = [folderName '_' num2str(dpd_params.learning_rate) '_'];
% folderName = [folderName 'Nup' num2str(roundn(nmse_up,-2)) '_'];
% folderName = [folderName 'Aup' num2str(roundn(acpr_up,-2))];

roadName = 'Result';
mkdir(roadName, folderName);
writetable(result_table,[roadName '\' folderName '\result_table.csv'])
save([roadName '\' folderName '\dpd_params.mat']', 'dpd_params')
save([roadName '\' folderName '\tx_data.mat']', 'tx_data')
save([roadName '\' folderName '\nmse_pa.mat']', 'nmse_pa')
result_history = dpd_sp.result_history;
save([roadName '\' folderName '\result_history.mat']', 'result_history')

% 保存图窗
all_fig = findall(groot,'Type','figure');
savefig(all_fig, [roadName '\' folderName '\allFiguresFile.fig'])

disp('complete')

