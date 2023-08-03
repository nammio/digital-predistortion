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
dpd_params.nIterations = 1;
dpd_params.learning_rate = 1;
dpd_params.learning_method = 'newton'; % Or 'ema' for exponential moving average.
dpd_params.use_dc_term = 1; % Adds an additional term for DC
% DDR参数
dpd_params.model_select = 'ddr';
dpd_params.use_simplifier = 1;
% GMP参数
dpd_params.use_even = false; 
dpd_params.use_conj = 0;    % Conjugate branch. Currently only set up for MP (lag = 0)
dpd_params.lag_depth = 2;  % 0 is a standard MP. >0 is GMP.
% 产生3种DPD
dpd = ILA_DPD(dpd_params);
dpd_sp = ILA_DPD(dpd_params);
dpd_omp = ILA_DPD(dpd_params);


%% Run Expierement: SP扫描
% 对照试验(1) 测试PA 
[y_pa, y_pa_orginal] = board.transmit(tx_data.data);
nmse_pa = dpd_sp.cal_nrmse(y_pa, tx_data.data, 'dB');
disp(['nmse without dpd is: ' num2str(nmse_pa) 'dB'])
acpr_pa = dpd_sp.cal_acpr(y_pa_orginal.measure_all_powers);
disp(['acpr without dpd is: ' num2str(acpr_pa) 'dB'])

maxIternation = 10;
spar = (5:22)';
% nmse_up_omp = zeros(size(spar));
% nmse_up_sp = zeros(size(spar));
% acpr_up_omp = zeros(size(spar));
% acpr_up_sp = zeros(size(spar));

RoadName = 'Result\compare_omp_sp\';
tableName = 'running_result_table.csv';

varNames = ["sparsity" "nmse_up_omp" "nmse_up_sp" "acpr_up_omp" "acpr_up_sp"];
% result_table = table(spar, nmse_up_omp, nmse_up_sp, acpr_up_omp, acpr_up_sp,'VariableNames',varNames);
% writetable(result_table,[RoadName tableName])

result_table = readtable([RoadName tableName]);
nmse_up_list = result_table.nmse_up_omp;
start_index = find(nmse_up_list == 0, 1, 'first');
% 性能对比
for i = start_index:18
    spar = i+4;
    [n_coeffs_sp,count] = dpd_sp.SP_learning(tx_data.data,board, spar,maxIternation);
    [y_sp, y_sp_orginal] = board.transmit(dpd_sp.predistort(tx_data.data));
    nmse_sp = dpd_sp.cal_nrmse(y_sp, tx_data.data, 'dB');
    acpr_sp = dpd.cal_acpr(y_sp_orginal.measure_all_powers);
    nmse_up_sp = nmse_sp - nmse_pa; acpr_up_sp = acpr_sp - acpr_pa;
    dpd_sp.reset_dpd
    
    n_coeffs = dpd_omp.omp_learning(tx_data.data, board, spar);
    [y_omp, y_omp_orginal]=board.transmit(dpd_omp.predistort(tx_data.data));
    nmse_omp = dpd_omp.cal_nrmse(y_omp, tx_data.data, 'dB');
    acpr_omp = dpd.cal_acpr(y_omp_orginal.measure_all_powers);
    nmse_up_omp = nmse_omp - nmse_pa; acpr_up_omp = acpr_omp - acpr_pa;
    dpd_omp.reset_dpd
    
    newInsert = [spar, nmse_up_omp, nmse_up_sp, acpr_up_omp, acpr_up_sp] ;
    result_table{i,:} = newInsert;
    writetable(result_table, [RoadName tableName]);
end

nmse_history = dpd_sp.result_history.nmse;
support_history = dpd_sp.result_history.support;
process = [nmse_history; support_history];
disp('complete')

