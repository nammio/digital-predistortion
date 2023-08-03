classdef ILA_DPD < handle
    %ILA_DPD. Inderect Learning Architecture DPD.
    %
    %  x(n)   +-----+ u(n) +-----+
    % +-----> | DPD +--+-> | PA  +------+
    %         +-----+  v   +-----+      |
    %                  --------+ e(n)   | y(n)
    %                  ^       |        |
    %                  |   +---v-+      |
    %                  +---+ DPD | <----+
    %               z(n)   +-----+
    %
    %
    %  MP DPD:
    %                +-----+    +---------------+
    %           +---->.    +---->b_1,1 ... b_1,M+-------+
    %           |    +-----+    +---------------+       |
    %           |                                       |
    %  x(n)     |    +-----+    +---------------+    +--v-+
    % +-------------->.|.|2+---->b_3,1 ... b_3,M+---->SUM +-------->
    %           |    +-----+    +---------------+    +--+-+
    %           |       .               .               ^
    %           |       .               .               |
    %           |       .                               |
    %           |    +-------+  +---------------+       |
    %           +---->.|.|p|1|  |b_p,1 ... b_p,M+-------+
    %                +-------+  +---------------+
    %
    %	Author:	Chance Tarver (2018)
    %		tarver.chance@gmail.com
    %
    
    properties
        order         % Nonlinear order of model. Can only be odd.
        use_even      % Include even order terms? true or false
        memory_depth  % Memory depth on each branch of the parallel hammerstein model
        lag_depth     % Memory depth of the lead/lag term in GMP
        nIterations   % Number of iterations used in the ILA learning
        coeffs        % DPD coefficients
        use_conj      % Use a conjugate branch as well
        use_dc_term   % use a dc term
        learning_rate % How much of the new iteration to use vs previous iteration. Should be in (0, 1]
        learning_method % Newton or ema
        coeff_history % Holds onto the coeffs used at each iteration
        result_history % Holds intermediate ACLR for each point during training in case of divergence.
        use_simplifier
        model_select
    end
    
    methods
        function obj = ILA_DPD(params)
            %ILA_DPD. Make a DPD module
            
            if nargin == 0
                params.order = 7;
                params.memory_depth = 3;
                params.lag_depth = 0;
                params.nIterations = 3;
                params.use_conj = 0;
                params.use_dc_term = 0;
                params.learning_rate = 0.75;
            end
            
            if mod(params.order, 2) == 0
                error('Order of the DPD must be odd.');
            end
            
            obj.order = params.order;
            obj.memory_depth = params.memory_depth;
            obj.use_dc_term = params.use_dc_term;
            obj.learning_rate = params.learning_rate;
            obj.nIterations = params.nIterations;
            obj.learning_method = params.learning_method;

            if isfield(params, 'model_select')
                % 设置二阶ddr模型参数
                obj.model_select = params.model_select;
            else
                obj.model_select = 'gmp';
            end
            
            switch obj.model_select
                case 'ddr'
                    obj.use_simplifier = params.use_simplifier;

                    % Start DPD coeffs being completely linear (no effect)
                    assert(obj.order >= 5, '2-nd ddr model must det order >= 5')

                    k_max = (obj.order - 1)/2;
                    n_coeffs = (k_max+1)*(obj.memory_depth+1) + k_max*obj.memory_depth;
                
                    if obj.use_simplifier
                        n_coeffs = n_coeffs + k_max*obj.memory_depth + ...
                            k_max*obj.memory_depth;
                    else
                        n_coeffs = n_coeffs + k_max*(obj.memory_depth+1)*obj.memory_depth/2 ...
                            + k_max*obj.memory_depth^2 + (k_max-1)*(obj.memory_depth+1)*obj.memory_depth/2;
                    end
            
                case 'gmp'
                    % 设置gmp模型参数
                    obj.use_even = params.use_even;
                    obj.lag_depth = params.lag_depth;
                    obj.use_conj = params.use_conj;

                    % Start DPD coeffs being completely linear (no effect)
                    if obj.use_even
                        assert(obj.lag_depth == 0, 'GMP not yet supported for even terms. Set lag_depth=0');
                        n_coeffs = obj.order * obj.memory_depth;
                    else
                        n_coeffs = obj.convert_order_to_number_of_coeffs * obj.memory_depth + ...
                            2*((obj.convert_order_to_number_of_coeffs-1) * obj.memory_depth * obj.lag_depth);
                    end

                    if obj.use_conj
                        n_coeffs = 2*n_coeffs;
                    end
            end
            
            if obj.use_dc_term
                n_coeffs = n_coeffs + 1;
            end 
            
            obj.coeffs = zeros(n_coeffs, 1);
            obj.coeffs(1) = 1;

        end
        
        
        function number_of_coeffs =  perform_learning(obj, x, pa)
            %perform_learning. Perfrom ILA DPD.
            %
            % The PA output is the input to the postdistorter used for
            % learning. We want the error to be zero which happens when the
            % ouput of the pre and post distorters are equal. So we need:
            %
            %     e = 0
            % u - z = 0
            %     u = z
            %     u = Y * beta
            %
            % We can set this up as a least squares regression problem.
            
            obj.coeff_history = obj.coeffs;
            obj.result_history.power  = zeros(3, obj.nIterations+1);
            obj.result_history.nmse = zeros(1, obj.nIterations);
            for iteration = 1:obj.nIterations
                % Forward through Predistorter
                u = obj.predistort(x);
                [y, test_signal] = pa.transmit(u); % Transmit the predistorted pa input
                obj.result_history.power(:, iteration) = test_signal.measure_all_powers;
                % Learn on postdistrter
                Y = setup_basis_matrix(obj, y);
                switch obj.learning_method
                    case 'newton'
                        post_distorter_out = obj.predistort(y);
                        error = u - post_distorter_out;
                        ls_result = ls_estimation(obj, Y, error);
                        % 计算LS估计的误差
                        ls_bias = obj.cal_nrmse(Y * ls_result, error, 'dB');
                        disp(['第' num2str(iteration) '次迭代中，LS估计的误差为' num2str(ls_bias) 'db']);
                        obj.result_history.nmse(iteration) = ls_bias;
                        % 更新系数
                        obj.coeffs = obj.coeffs + (obj.learning_rate) * ls_result;
                    case 'ema'
                        ls_result = ls_estimation(obj, Y, u);
                        obj.coeffs = (1-obj.learning_rate) * obj.coeffs + (obj.learning_rate) * ls_result; % 更新系数
                        % 计算LS估计的误差
                        ls_bias = obj.cal_nrmse(Y * ls_result, u, 'dB');
                        disp(['第' num2str(iteration) '次迭代中，LS估计的误差为' num2str(ls_bias) 'd']);
                        obj.result_history.nmse(iteration) = ls_bias;
                end
                obj.coeff_history = [obj.coeff_history obj.coeffs];
            end
            % Need extra to evaluate final iteration
            u = obj.predistort(x);
            [~, test_signal] = pa.transmit(u); % Transmit the predistorted pa input
            obj.result_history.power(:, iteration+1) = test_signal.measure_all_powers;
            number_of_coeffs = numel(obj.coeffs(obj.coeffs~=0));
        end
        
        
        function beta = ls_estimation(obj, X, y)
            %ls_estimation
            % Solves problems where we want to minimize the error between a
            % lienar model and some input/output data.
            %
            %     min || y - X*beta ||^2
            %
            % A small regularlizer, lambda, is included to improve the
            % conditioning of the matrix.
            %
            
            % Trim X and y to get rid of 0s in X.
            switch obj.model_select
                case 'gmp'
                    X = X(obj.memory_depth+obj.lag_depth:end-obj.lag_depth, :);
                    y = y(obj.memory_depth+obj.lag_depth:end-obj.lag_depth);
                case 'ddr'
                    X = X(1+obj.memory_depth:end, :);
                    y = y(1+obj.memory_depth:end);
            end
            lambda = 2^(-16);
            beta = (X'*X + lambda*eye(size((X'*X)))) \ (X'*y);
        end
        
        function X = setup_basis_matrix(obj, x)
            switch obj.model_select
                case 'gmp'
                    X = obj.gmp_basis_matrix(x);
                case 'ddr'
                    X = obj.ddr_basis_matrix(x);
            end
        end
        
        function X = gmp_basis_matrix(obj, x)
            %gmp_basis_matrix. Setup the basis matrix(gmp) for the LS learning of
            %the PA parameters or for broadcasting through the PA model.
            %
            % obj.gmp_basis_matrix(x)
            %
            % Inputs:
            %   x - column vector of the PA input signal.
            % Output:
            %   X - matrix where each column is the signal, delayed version of
            %   a signal, signal after going through a nonlinearity, or both.
            %
            %	Author:	Chance Tarver (2018)
            %		tarver.chance@gmail.com
            %
            
            number_of_basis_vectors = numel(obj.coeffs);
            X = zeros(length(x), number_of_basis_vectors);
            
            if obj.use_even
                step_size = 1;
            else
                step_size = 2;
            end
            
            % Main branch
            count = 1;
            for i = 1:step_size:obj.order
                branch = x .* abs(x).^(i-1);
                for j = 1:obj.memory_depth
                    delayed_version = zeros(size(branch));
                    delayed_version(j:end) = branch(1:end - j + 1);
                    X(:, count) = delayed_version;
                    count = count + 1;
                end
            end
            
            % Lag term
            for k = 3:step_size:obj.order  % Lag/Lead doesn't exist for k=1
                absolute_value_part_base = abs(x).^(k-1);
                for m = 1:obj.lag_depth
                    lagged_abs = [zeros(m,1); absolute_value_part_base(1:end-m)];
                    main_base = x .* lagged_abs;
                    for l = 1:obj.memory_depth
                        X(l:end, count) = main_base(1:(end-l+1));
                        count = count + 1;
                    end
                end
            end
            
            % Lead term
            for k = 3:step_size:obj.order  % Lag/Lead doesn't exist for k=1
                absolute_value_part_base = abs(x).^(k-1);
                for m = 1:obj.lag_depth
                    lead_abs = [absolute_value_part_base(1+m:end); zeros(m,1)];
                    main_base = x .* lead_abs;
                    for l = 1:obj.memory_depth
                        X(l:end, count) = main_base(1:(end-l+1));
                        count = count + 1;
                    end
                end
            end
            
            if obj.use_conj
                % Conjugate branch
                for i = 1:step_size:obj.order
                    branch = conj(x) .* abs(x).^(i-1);
                    for j = 1:obj.memory_depth
                        delayed_version = zeros(size(branch));
                        delayed_version(j:end) = branch(1:end - j + 1);
                        X(:, count) = delayed_version;
                        count = count + 1;
                    end
                end
            end
            
            % DC
            if obj.use_dc_term
                X(:, count) = 1;
            end
        end
        
        
        function number_of_coeffs = convert_order_to_number_of_coeffs(obj, order)
            %convert_order_to_number_of_coeffs. Helper function to easily
            %convert the order to number of coeffs. We need this because we
            %only model odd orders.
            
            if nargin == 1
                order = obj.order;
            end
            
            number_of_coeffs = (order + 1) / 2;
        end
        
        
        function out = predistort(obj, x)
            %predistort. Use the coeffs stored in object to predistort an
            %input.
            X = obj.setup_basis_matrix(x);
            out = X * obj.coeffs;
        end
        
        
        function plot_history(obj)
            % plot_history. Plots how the magnitude of the DPD coeffs
            % evolved over each iteration.
            figure(55);
            iterations = 0:obj.nIterations;
            subplot(1,2,1)
            hold on;
            plot(iterations, abs(obj.coeff_history'));
            title('History for DPD Coeffs Learning');
            xlabel('Iteration Number');
            ylabel('abs(coeffs)');
            grid on;
            subplot(1,2,2)
            
            plot(iterations, (obj.result_history.power'));
            grid on;
            title('Performance vs Iteration');
            ylabel('dBm');
            xlabel('Iteration Number');
            legend('L1', 'Main Channel', 'U1', 'Location', 'best')
        end
                function X = ddr_basis_matrix(obj, x)
            % 设置二阶DDR模型基函数矩阵
            
            number_of_basis_vectors = numel(obj.coeffs);
            X = zeros(length(x), number_of_basis_vectors);
            
            count = 1;
            
            % setup 1-nd term-1
            for k = 0:(obj.order - 1)/2
                branch = abs(x).^(2*k);
                for i = 0:obj.memory_depth
                    delay_version = zeros(size(x));
                    delay_version(1+i:end,:) = x(1:end-i,:);
                    bf_temp = branch.*delay_version;
                    X(:,count) = bf_temp;
                    count = count+1;
                end
            end
            
            % setup 1-nd term-2
            for k = 1:(obj.order - 1)/2
                branch = abs(x).^(2*k-2) .* (x.^2);
                for i = 1:obj.memory_depth
                    delay_version = zeros(size(x));
                    delay_version(1+i:end,:) = conj(x(1:end-i,:));
                    bf_temp = branch .* delay_version;
                    X(:,count) = bf_temp;
                    count = count+1;
                end
            end
            
            if obj.use_simplifier == false
            
                % setup 2-nd term-1
                for k = 1:(obj.order - 1)/2
                    branch = (abs(x).^(2*k-2)) .* conj(x);
                    for i1 = 1:obj.memory_depth
                        delay_version1 = zeros(size(x));
                        delay_version1(1+i1:end,:) = x(1:end-i1,:);
                        for i2 = i1:obj.memory_depth
                            delay_version2 = zeros(size(x));
                            delay_version2(1+i2:end,:) = x(1:end-i2,:);
                            bf_temp = branch.*(delay_version1.*delay_version2);
                            X(:,count) = bf_temp;
                            count = count+1;
                        end
                    end
                end

                % setup 2-nd term-2
                for k = 1:(obj.order - 1)/2
                    branch = abs(x).^(2*k-2) .* x;
                    for i1 = 1:obj.memory_depth
                        delay_version1 = zeros(size(x));
                        delay_version1(1+i1:end,:) = conj(x(1:end-i1,:));
                        for i2 = 1:obj.memory_depth
                            delay_version2 = zeros(size(x));
                            delay_version2(1+i2:end,:) = x(1:end-i2,:);
                            bf_temp = branch.*(delay_version1.*delay_version2);
                            X(:,count) = bf_temp;
                            count = count+1;
                        end
                    end
                end 

                % setup 2-nd term-3
                for k = 2:(obj.order - 1)/2
                    branch = abs(x).^(2*k-4) .* (x.^3);
                    for i1 = 1:obj.memory_depth
                        delay_version1 = zeros(size(x));
                        delay_version1(1+i1:end,:) = conj(x(1:end-i1,:));
                        for i2 = i1:obj.memory_depth
                            delay_version2 = zeros(size(x));
                            delay_version2(1+i2:end,:) = conj(x(1:end-i2,:));
                            bf_temp = branch.*(delay_version1.*delay_version2);
                            X(:,count) = bf_temp;
                            count = count+1;
                        end
                    end
                end 
            
            else
                % setup simplified 2-nd term-1
                for k = 1:(obj.order - 1)/2
                    branch = abs(x).^(2*k-2) .* x;
                    for i = 1:obj.memory_depth
                        delay_version = zeros(size(x));
                        delay_version(1+i:end,:) = abs(x(1:end-i,:)).^2;
                        bf_temp = branch .* delay_version;
                        X(:,count) = bf_temp;
                        count = count+1;
                    end
                end

                % setup simplified 2-nd term-2
                for k = 1:(obj.order - 1)/2
                    branch = abs(x).^(2*k-2) .* conj(x);
                    for i = 1:obj.memory_depth
                        delay_version = zeros(size(x));
                        delay_version(1+i:end,:) = x(1:end-i,:).^2;
                        bf_temp = branch .* delay_version;
                        X(:,count) = bf_temp;
                        count = count+1;
                    end
                end 
            end
            
            % DC
            if obj.use_dc_term
                X(:, count) = 1;
            end
            
        end
        
        function acpr = cal_acpr(~, power_dbm)
            % 给定主信道与邻信道功率（单位dbm），计算acpr
            l1_power = (10^(power_dbm(1)/10))*0.001;
            main_power = (10^(power_dbm(2)/10))*0.001;
            u1_power = (10^(power_dbm(3)/10))*0.001;
            acpr = 10*log10((l1_power+u1_power) / (2*main_power));
        end

        
        function out = cal_nrmse(~, x, y, token)
            % 计算NRMSE，x是实际信号，y是理想信号
            % 不设置token参数，函数计算nrmse;将token设置为'db',函数计算nmse
            assert(length(x) == length(y), '输入向量size不同')
            
            x_rms = rms(x);
            y_rms = rms(y);
            nrmse = rms(y/y_rms - x/x_rms);
            nmse_db = 20*log10(nrmse);
            
            if nargin == 3 
                out = nrmse * 100;
            else
                if token ~= false
                    out = nmse_db;
                else
                    error('nrmse error');
                end
            end
        end
    
        function beta_omp = omp(obj, X, y, sparsity)
            %omp算法
            % 对X的列向量归一化
            col_norm_X = zeros(size(X));
            for i = 1:size(X,2)
                scale_factor = norm(X(:,i));
                col_norm_X(:,i) = X(:,i)./scale_factor;
            end
            % 初始化
            % lambda = 2^(-16);
            bias = y;
            bf_index = [];
            beta_omp = zeros(size(X,2),1);
            for i = 1:sparsity
                cor = abs(col_norm_X' * bias);
                [~, ind] = max(cor);
                bf_index = union(bf_index, ind);
                bf_index = unique(bf_index, 'stable');
                X_temp = X(:,bf_index);
                beta = obj.ls_estimation(X_temp, y);
                % beta = ((X_temp'*X_temp) + lambda*eye(length(bf_index)))\ (X_temp'*y);
                bias = y - X_temp * beta; 
            end
            beta_omp(bf_index) = beta;
        end  
        
        
        function number_of_coeffs = omp_learning(obj, x, pa, spar)
            u = obj.predistort(x);
            y = pa.transmit(u);
            Y = obj.setup_basis_matrix(y);
            obj.result_history.support_set = zeros(spar,1);
            bias = u;
            for i = 1:spar
                index_star = obj.select_index(Y, bias, 1);
                obj.result_history.support_set(i,1) = index_star;
                support_set_temp = obj.result_history.support_set(obj.result_history.support_set ~= 0);
                support_set_temp = unique(support_set_temp, 'stable');
                Y_support = Y(:, support_set_temp);
                ls_result = obj.ls_estimation(Y_support, u);
                obj.coeffs(support_set_temp) = ls_result;
                post_predistort = obj.predistort(y);
                bias = u - post_predistort;
            end
            number_of_coeffs =  numel(obj.coeffs(obj.coeffs~=0));
        end
      
        
        function index = select_index(~, X, y, k)
            % 选择相关度最高的X中的列
            column_norm_X = zeros(size(X));
            for i = 1:size(X,2)
                scale_factor = norm(X(:,i));
                column_norm_X(:,i) = X(:,i)./scale_factor;
            end
            cor = abs(column_norm_X' * y);
            [~, index] = maxk(cor,k);
        end
            
        function number_of_coeffs = perform_omp_training(obj, x, pa, spar)
            obj.coeff_history = obj.coeffs;
            obj.result_history.power  = zeros(3, obj.nIterations+1);
            obj.result_history.nmse = zeros(1, obj.nIterations);
            for iteration = 1:obj.nIterations
                % Forward through Predistorter
                u = obj.predistort(x);
                [y, test_signal] = pa.transmit(u); % Transmit the predistorted pa input
                obj.result_history.power(:, iteration) = test_signal.measure_all_powers;
                % Learn on postdistrter
                Y = setup_basis_matrix(obj, y);
                ls_result = omp(obj, Y, u, spar);
                obj.coeffs = (1-obj.learning_rate) * obj.coeffs + (obj.learning_rate) * ls_result;
                obj.coeffs = ls_result;
                % 计算LS估计的误差
                ls_bias_nmse = obj.cal_nrmse(Y * ls_result, u, 'dB');
                disp(['第' num2str(iteration) '次迭代中，LS估计的误差为' num2str(ls_bias_nmse) 'dB']);
                obj.result_history.nmse(iteration) = ls_bias_nmse;
                
                obj.coeff_history = [obj.coeff_history obj.coeffs];

            end
            % Need extra to evaluate final iteration
            u = obj.predistort(x);
            [~, test_signal] = pa.transmit(u); % Transmit the predistorted pa input
            obj.result_history.power(:, iteration+1) = test_signal.measure_all_powers;
            number_of_coeffs = numel(obj.coeffs(obj.coeffs~=0));
        end
        
        
        function [number_of_coeffs,count] = SP_learning(obj, x, pa, spar, intv)
            if nargin == 4 || intv == 0
                maxIteration = obj.nIterations;
                syn_intv = 0;
            else
                maxIteration = intv * obj.nIterations - 1;
                syn_intv = 0:intv:maxIteration;
            end
            obj.result_history.nmse = zeros(1, maxIteration);
            obj.result_history.support = zeros(spar,maxIteration);
            u = obj.predistort(x);
            y = pa.transmit(u);
            Y = obj.setup_basis_matrix(y);
            bias = u ;
            support_set  = [];
            count = 0;
            for i = 1:maxIteration
                count = count + 1;
                %(1) Identification
                index_star = obj.select_index(Y, bias, spar);
                %(2) Support Merger
                support_set_temp = union(support_set, index_star);
                support_set_temp = unique(support_set_temp);
                Y_support_temp = Y(:, support_set_temp);
                %(3) 求LS解
                theta_temp = obj.ls_estimation(Y_support_temp, u);          
                %(4) Pruning
                theta_temp_norm = theta_temp./sqrt(diag((Y_support_temp'*Y_support_temp)));
                [~, pos_star] = sort(abs(theta_temp_norm), 'descend');
                %(5) Updata Support
                support_set = support_set_temp(pos_star(pos_star(1:spar)));
                Y_support = Y(:, support_set);
                %(6) Re-LS
                theta = obj.ls_estimation(Y_support, u);
                %(7) Updata bias
                if sort(syn_intv == i)
                    obj.reset_dpd; obj.coeffs(1) = 0;
                    obj.coeffs(support_set) = theta;
                    u = obj.predistort(x);
                    y = pa.transmit(u);
                    Y = obj.setup_basis_matrix(y);
                else
                    bias = u - Y_support * theta;
                end
                obj.result_history.nmse(1,i) = obj.cal_nrmse(Y_support*theta, u);
                obj.result_history.support(:,i) = support_set;
                %(8) Stop condition
                if obj.result_history.nmse(i) < -8
                    obj.coeffs(support_set) = theta;
                    break
                else
                    if i>1 && obj.result_history.nmse(i) > obj.result_history.nmse(i-1)
                        % break
                    end
                end
            end
            obj.reset_dpd; obj.coeffs(1) = 0;
            obj.coeffs(support_set) = theta;
            obj.result_history.nmse = obj.result_history.nmse(1,1:count);
            number_of_coeffs = numel(obj.coeffs(obj.coeffs~=0));
        end
        
        function reset_dpd(obj)
            obj.coeffs(1) = 1;
            obj.coeffs(2:end) = 0;
        end
        
    end
end
