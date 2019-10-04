% run_DYNMONGA
% fitness_1: Q(����ǰ��); fitness_2: NMI
% ����cellphone, enron mail���ݼ�
% clear all;
% clc;

%�˹�����
% load('datasets/syn_fix_3.mat');
% load('datasets/syn_fix_5.mat');
% load('datasets/syn_var_3.mat');
% load('datasets/syn_var_5.mat');

%��ʵ����
% load('datasets/cell.mat');
% load('result/firststep_cluster_cell.mat');
% load('datasets/enron.mat');
% load('result/firststep_cluster_enron.mat');
% GT_Cube = dynMoeaResult;


%%
% dynMod = [];
% dynNmi = [];
% dynPop = {};
% dynTime = [];
% dynMONGA_Result = {};

% M = 2;                            %M-objective Problems
% pop_size = 100;
% maxgen = 100;                     %Maximum number of generations
% neighbor_num = 10;                 %��decomposed approach�й�
% % crossover_rate = ��;            %�Ӵ�ÿ����Ⱦɫ����н���
% mutation_rate = 0.20;             %�Դ�������0.005Ҳ�ǿ��Խ��ܵģ����������ռ�ܱ����ı���
% children_proportion = 1;       %ÿ���Ŵ����������Ӵ�ռ���������ı���
% Threshold = 0.80;                 %PM����
% whole_size = ceil((1+children_proportion)*pop_size); %��������Ժ��Ⱦɫ������
% p_migration = 0.50;

for r = 5
    %Tchebycheff approach
    global idealp weights neighbors;
    % idelp �ǲο��� (z1, z2)��z1��z2�ֱ��Ǹ�Ŀ�꺯������ֵ
    num_time_stamp = size(W_Cube, 2); %W_Cube contains 10 cells restoring temporal adjacent matrices
    
    num_iter = 1;
    % time step is 1;
    
    %real-world network
    [Mod,chromosome,cluster,time] = PNGACD_Decomposed(W_Cube{1},Threshold); %�Ƚ����ٱ���orǨ��
    
    dynMONGA_Result{1,r} = cluster; %�����ŵ����绮��
    dynMod(1,r) = Mod;
%     dynNmi(1,r) = NMI(GT_Matrix(:,1)',cluster); %�˹�����GT_Matrix
    dynNmi(1,r) = NMI(GT_Cube{num_iter},cluster); %������ʵ����
    dynPop{1,r} = chromosome;
    dynTime(1,r) = time;
    disp(['time_stamp = ', num2str(num_iter), ', Q = ', ...
        num2str(Mod), ', NMI = ', num2str(dynNmi(1,r))])  
%     disp(['time_stamp = ', num2str(num_iter), ', Q = ', num2str(Mod)])
    
    %%
    for num_iter = 2 : num_time_stamp
        tic;
        
        adj_mat = W_Cube{num_iter};
        if isequal(adj_mat, adj_mat') == 0 %�ж��Ƿ�Ϊ�ԳƵ��ڽӾ���
            adj_mat = adj_mat + adj_mat';
        end
        % �ڽӾ���Խ���Ԫ��Ϊ0
        [row] = find(diag(adj_mat)); %�ҵ�nonzero�ĶԽ���Ԫ��id
        for id = row
            adj_mat(id,id) = 0;
        end
        
        node_num = size(adj_mat,2);       %node number
        edge_num = sum(sum(adj_mat))/2;   %edge number
        [edge_begin_end] = CreatEdgeList(adj_mat);
        %     edgelist = edges_list(adj_mat,node_num); %��¼ÿ��������ߣ�����
        adjacent_num = round(0.05*node_num);  %��Ⱥ��ʼ�������Ľڵ������������Խڵ�����0.1�������ˣ���ʵ���㷨�Դ˲���������
        child_chromosome = struct('genome',{},'clusters',{},'fitness_1',{},'fitness_2',{});
        
        
        %% ��ʼ��
        % initialize EP
        EP = [];
        % initialize z*
        idealp = -Inf * ones(1,M);  %�ο���(z1,z2)
        % find s neighbor solutions to each individual
        [weights, neighbors] = init_weight(pop_size,neighbor_num);
        % initalize chromosome
        [chromosome] = PCDInitial_Population(pop_size,adj_mat,adjacent_num,node_num,Threshold); %PM��ʼ����
        % compute ���Ŀ�꺯��ֵ
        [chromosome] = evaluate_objectives(chromosome, pop_size, node_num, edge_num, adj_mat, dynMONGA_Result{num_iter-1,r}); % �������ǰ�����ģ��ȣ�NMI
        
        %%
        % ��ʼ���Ժ���idealpoint z*
        f = [];
        for j = 1 : pop_size
            f = [f; chromosome(j).fitness_1];
            idealp = min(f);
        end
        
        %% compute snapshot and temporal costs
        % ������ʼ
        
        for t = 1 : maxgen  %��t�ε���
            % �ȱ����ٽ���
            
            %��ÿ�������⣬��neighbors��ѡ�б���ĸ��壬ѡ�еĸ���Ҳ���ڽ���
            for pop_id = 1 : pop_size
                selected_neighbor_id = [];
                selected_pop_id = [];
                
                %%
                %neighbors��ѡ��"��ͬ"��������(��һ��ѡ���Լ�)
                while isempty(selected_neighbor_id) || selected_neighbor_id(1) == selected_neighbor_id(2);
                    selected_neighbor_id =  randi(neighbor_num, 1, 2);
                end
                selected_pop_id = neighbors(pop_id, selected_neighbor_id);
                %%
                % %neighbors��ѡ��������(ѡ���Լ� + һ���ھ�)
                % selected_neighbor_id(1,1) = 1;
                % selected_neighbor_id(1,2)=  randi(neighbor_num-1, 1) + 1; %id=1���ھ����Լ�
                % selected_pop_id = neighbors(pop_id, selected_neighbor_id);
                
                %%
                %reproduce a child population
                child_chromosome(pop_id) = DeCross_Over(chromosome, selected_pop_id, node_num);
                if rand(1) < p_migration
                    child_chromosome(pop_id) = DeMutation(child_chromosome(pop_id), mutation_rate, edge_begin_end);
                else
                    %migration
                    child_chromosome(pop_id) = DeMigration(child_chromosome(pop_id), node_num, adj_mat);
                end
                
                %%
                % evaluate_objectives��Q��NMIΪ��
                child_chromosome(pop_id) = evaluate_objectives(child_chromosome(pop_id), 1, node_num, ...
                    edge_num, adj_mat, dynMONGA_Result{num_iter-1,r}); % �������ǰ�����ģ��ȣ�NMI
                
                %%
                %���ڷֽ�Ķ�Ŀ���Ż�
                %���²������Ӵ����ھӸ������бȽϣ� ����fitness�Ƚ�global_fitС��Ⱦɫ�嵽neighbors��
                %ÿ���Ӵ������������Ժ����
                %update neighbors (g(pop_id))
                for k = neighbors(pop_id,:)
                    child_fit = decomposedFitness(weights(k,:), child_chromosome(pop_id).fitness_1, idealp);
                    gbest_fit = decomposedFitness(weights(k,:), chromosome(k).fitness_1, idealp);
                    if child_fit < gbest_fit
                        chromosome(k).genome = child_chromosome(pop_id).genome;
                        chromosome(k).clusters = child_chromosome(pop_id).clusters;
                        chromosome(k).fitness_1 = child_chromosome(pop_id).fitness_1;
                        chromosome(k).fitness_2 = child_chromosome(pop_id).fitness_2;
                    end
                end
                
            end
            
            %%
            % ȫ��offspring�����Ժ��ٸ�����neighbors
            for pop_id = 1 : pop_size
                
                % ��֧������ ���� ����ǰģ��� + ������NMI
                if isempty(EP)
                    EP = [EP chromosome(pop_id)];
                else
                    isDominate = 0;
                    isExist = 0;
                    rmindex = [];
                    for k = 1 : numel(EP) % numel����Ԫ������
                        if isequal(chromosome(pop_id).clusters, EP(k).clusters)  % isequal(chromosome(pop_id).genome, EP(k).genome)
                            isExist = 1;
                        end
                        if dominates(chromosome(pop_id), EP(k))
                            rmindex = [rmindex k];
                        elseif dominates(EP(k), chromosome(pop_id))
                            isDominate = 1;
                        end
                    end
                    EP(rmindex) = [];
                    if ~isDominate && ~isExist
                        EP = [EP chromosome(pop_id)];
                    end
                end
                
                %update idealp
                idealp = min([child_chromosome(pop_id).fitness_1; idealp]);
            end
            
        end
        
        time = toc;
        
        % %����ģ����ܶ�
        % density = [];
        % for front = EP
        %     density = [density; modularity_density(adj_mat, front.clusters)];
        % end
        % [EP] = Sort_density(EP, density, size(EP,2)); %����ģ����ܶȽ��н�������
        % % [~,index] = max(density); %ģ����ܶ�����Ⱦɫ��
        
        Modularity = [];
        for front = EP
            Modularity = [Modularity; abs(front.fitness_2(1))];
        end
        [~,index] = max(Modularity);
        dynMONGA_Result{num_iter,r} = EP(index).clusters;  %�����ŵ����绮��
        dynMod(num_iter,r) = -EP(index).fitness_2(1); %������ ������ ��ģ���
%         dynNmi(num_iter,r) = NMI(EP(index).clusters, GT_Matrix(:,num_iter)'); %�˹�����
        dynNmi(num_iter,r) = NMI(EP(index).clusters, GT_Cube{num_iter});  %��ʵ��������
        dynPop{num_iter,r} = chromosome;
        dynTime(num_iter,r) = time;
        disp(['time_stamp = ', num2str(num_iter), ', Q = ', ...
            num2str(-EP(index).fitness_2(1)), ', NMI = ', num2str(dynNmi(num_iter,r))]);
%         disp(['time_stamp = ', num2str(num_iter), ', Q = ', ...
%             num2str(-EP(index).fitness_2(1)), ', NMI = ', num2str(-EP(index).fitness_2(2))]);
        
    end
end


