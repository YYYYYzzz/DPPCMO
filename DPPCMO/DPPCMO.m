classdef DPPCMO < ALGORITHM
% <multi> <real/integer/label/binary/permutation> <constrained>
% Coevolutionary constrained multi-objective optimization framework
% type --- 1 --- Type of operator (1. GA 2. DE)


%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            type = Algorithm.ParameterSet(1);

            %% Generate random population
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            Fitness1    = CalFitness(Population1.objs,Population1.cons,0);
            Fitness2    = CalFitness(Population2.objs,Population2.cons,1e6);
            cp               = 2;
            change_threshold = 1e-3;
            state = 0;
            gen              = 1;
            Tmax             = ceil(Problem.maxFE/Problem.N);
            epsilon = +Inf;

            %% Optimization
            while Algorithm.NotTerminated2(Population1,Population2)
                Objvalues(gen) = sum(sum(Population2.objs,1));
                if type == 1
                    MatingPool1 = TournamentSelection(2,Problem.N,Fitness1);
                    MatingPool2 = TournamentSelection(2,Problem.N,Fitness2);
                    Offspring1  = OperatorGAhalf(Problem,Population1(MatingPool1));
                    Offspring2  = OperatorGAhalf(Problem,Population2(MatingPool2));
                elseif type == 2
                    MatingPool1 = TournamentSelection(2,2*Problem.N,Fitness1);
                    MatingPool2 = TournamentSelection(2,2*Problem.N,Fitness2);
                    Offspring1  = OperatorDE(Problem,Population1,Population1(MatingPool1(1:end/2)),Population1(MatingPool1(end/2+1:end)));
                    Offspring2  = OperatorDE(Problem,Population2,Population2(MatingPool2(1:end/2)),Population2(MatingPool2(end/2+1:end)));
                end
                
                if gen ~= 1
                   state = is_stable(Objvalues,gen,Population2,Problem.N,change_threshold,Problem.M);
                end
                if state == 1
                    if cp >= 0
                        epsilon_0 = max(sum(max(0,Population2.cons),2));
                        cp = -(log(epsilon_0)+6)/log(1-0.5);
                        epsilon = epsilon_0 * ((1 - (gen / Tmax)) ^ cp);
                        state = 0;
                    end
                end
                
                if epsilon == +Inf
                    Q1 = [Population1,Offspring1,Offspring2];
                    Q2 = [Population2,Offspring1,Offspring2];
                else
                    CV1 = sum(max(0,Population2.cons),2);
                    Feasible_rate = mean(CV1 == 0);
                    if Feasible_rate > 0
                        Next1(1:Problem.N) = false;
                        for i = 1:Problem.N
                            if CV1(i) == 0
                                Next1(i) = true;         % Migrate solution x into Population1
                             end
                        end
                        arch = Population2(Next1);
                        Q1 = [Population1,arch,Offspring1,Offspring2];
                        Q2 = [Population1,Population2,Offspring1,Offspring2];
                    else
                        Q1 = [Population1,Offspring1,Offspring2];
                        Q2 = [Population2,Offspring1,Offspring2];
                    end
                end                
                [Population1,Fitness1] = EnvironmentalSelection(Q1,Problem.N,0);
                [Population2,Fitness2] = EnvironmentalSelection(Q2,Problem.N,epsilon);              
                gen = gen + 1;
            end
        end
    end
end
function result = is_stable(Objvalues,gen,Population,N,change_threshold,M)
    result = 0;
    [FrontNo,~]=NDSort(Population.objs,size(Population.objs,1));
    NC=size(find(FrontNo==1),2);
    max_change = abs(Objvalues(gen)-Objvalues(gen-1));
    if NC == N
        change_threshold = change_threshold * abs(((Objvalues(gen) / N))/(M))*10^(M-2);
        if max_change <= change_threshold
            result = 1;
        end
    end
end