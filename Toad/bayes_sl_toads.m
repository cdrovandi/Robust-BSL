function [theta,loglike] = bayes_sl_toads(ssy,n,M,cov_rw,start,simArgs,sumArgs,NaN_Pos,sumstat_fun)
% running standard BSL method for Toad example
% INPUT:
% Y - toads moves observation matrix, dimension is ndays by ntoads
% n - number of simulated data sets for estimating synthetic likelihood
% M - number of MCMC iterations
% cov_rw - the covariance matrix used in the random walk (uses multivariate random normal for proposals)
% start - initial value of parameters for Markov chain
% simArgs - extra parameters required in simulation model
% sumArgs - extra parameters required in summary statistic function
% NaN_Pos - positions of missing values in the data
% sumstat_fun - summary statistic function
%
% OUTPUT:
% theta - MCMC samples of parameter values
% loglike - MCMC samples of estimated log likelihood

% extract extra simulation parameters
ntoads = simArgs.ntoads;
ndays = simArgs.ndays;
model = simArgs.model;
d0 = simArgs.d0;
lag = sumArgs.lag;


theta_curr = start; %Initial guesses for parameters
ns = length(ssy); % number of summary statistics
theta = zeros(M,3); % storing mcmc chain for parameters
loglike = zeros(M,1); % storing mcmc values for log likelihoods

% Simulating n sets of data and taking their summary statistics
ssx = zeros(n,ns);
parfor k = 1:n
    X = simulate_toads(theta_curr,ntoads,ndays,model,d0);
    X(NaN_Pos) = NaN;
    ssx(k,:) = feval(sumstat_fun,X,lag);
end

% Calculating the mean and covariance of the summary statistics
the_mean = mean(ssx);
the_cov = cov(ssx);

% estimate logdet numerically stably
L = chol(the_cov);
logdetA = 2*sum(log(diag(L)));

% synthetic likelihood
loglike_ind_curr = -0.5*logdetA - 0.5*(ssy-the_mean)/the_cov*(ssy-the_mean)';

for i = 1:M
    
    fprintf('i = %i\n',i)
    
    % Proposing new parameters (proposed on transformed space)
    theta_tilde_curr = para_transformation(theta_curr);
    theta_tilde_prop = mvnrnd(theta_tilde_curr,cov_rw);
    theta_prop = para_back_transformation(theta_tilde_prop); % transform back to original transformation
    prob = jacobian_transformation(theta_tilde_prop) / jacobian_transformation(theta_tilde_curr); % jacobian of transformation required in MH ratio
            
    % Simulating n sets of data and taking their summary statistics
    ssx = zeros(n,ns);
    parfor k = 1:n
        X = simulate_toads(theta_prop,ntoads,ndays,model,d0);
        X(NaN_Pos) = NaN;
        ssx(k,:) = feval(sumstat_fun,X,lag);
    end
    
    % Calculating the mean and covariance of the summary statistics
    the_mean = mean(ssx);
    the_cov = cov(ssx);
    
    % estimate logdet numerically stably
    L = chol(the_cov);
    logdetA = 2*sum(log(diag(L)));
    
    % synthetic likelihood
    loglike_ind_prop = -0.5*logdetA - 0.5*(ssy-the_mean)/the_cov*(ssy-the_mean)';

    % Metropolis-Hastings accept/reject
    if (prob * exp(loglike_ind_prop - loglike_ind_curr) > rand)
        fprintf('*** accept ***\n');
        theta_curr = theta_prop;
        loglike_ind_curr = loglike_ind_prop;
    end
    
    % store current values of the chain
    theta(i,:) = theta_curr;
    loglike(i) = loglike_ind_curr;   
    
end


end

