function [theta,loglike,gamma] = bayes_sl_toads_variance(ssy,n,M,cov_rw,start,simArgs,sumArgs,NaN_Pos,reg_mean,sumstat_fun)
% running BSL method with mean adjustment for Toad example
% INPUT:
% Y - toads moves observation matrix, dimension is ndays by ntoads
% n - number of simulated data sets for estimating synthetic likelihood
% M - number of MCMC iterations
% cov_rw - the covariance matrix used in the random walk (uses multivariate random normal for proposals)
% start - initial value of parameters for Markov chain
% simArgs - extra parameters required in simulation model
% sumArgs - extra parameters required in summary statistic function
% NaN_Pos - positions of missing values in the data
% reg_mean - parameter of the prior for the gamma parameters of the variance inflation
% sumstat_fun - summary statistic function
%
% OUTPUT:
% theta - MCMC samples of parameter values
% loglike - MCMC samples of estimated log likelihood
% gamma - MCMC samples of gamma parameter values of the variance inflation

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

gamma_curr = reg_mean*ones(1,ns); % initial value of gamma variance inflation parameters
gamma = zeros(M,ns); % store mcmc chain for gamma parameters

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
ssx_curr = ssx;
std_curr = std(ssx);
the_cov = the_cov + diag((std_curr.*gamma_curr).^2);

% estimate logdet numerically stably
L = chol(the_cov);
logdetA = 2*sum(log(diag(L)));

% synthetic likelihood
loglike_ind_curr = -0.5*logdetA - 0.5*(ssy-the_mean)/the_cov*(ssy-the_mean)';

for i = 1:M
    
    fprintf('i = %i\n',i)
    
    % update gamma using slice sampler
    the_cov_base = cov(ssx_curr);
    the_mean = mean(ssx_curr);
    for j = 1:ns
        lower = 0;
        target = loglike_ind_curr + sum(log(exppdf(gamma_curr,reg_mean))) - exprnd(1);
        
        % step out for upper limit
        curr = gamma_curr(j);
        upper = gamma_curr(j) + 1;
        while(1)
            gamma_upper = gamma_curr;
            gamma_upper(j) = upper;
            the_cov_upper = the_cov_base + diag((std_curr.*gamma_upper).^2);
            L = chol(the_cov_upper);
            logdetA = 2*sum(log(diag(L)));
            loglike_ind_upper = -0.5*logdetA - 0.5*(ssy-the_mean)/the_cov_upper*(ssy-the_mean)';
            target_upper = loglike_ind_upper + sum(log(exppdf(gamma_upper,reg_mean)));
            if (target_upper < target)
                break;
            end
            upper = upper + 1;
        end
        
        % shrink
        while(1)
            prop = unifrnd(lower,upper);
            gamma_prop = gamma_curr;
            gamma_prop(j) = prop;
            
            the_cov_prop = the_cov_base + diag((std_curr.*gamma_prop).^2);
            L = chol(the_cov_prop);
            logdetA = 2*sum(log(diag(L)));
            loglike_ind_prop = -0.5*logdetA - 0.5*(ssy-the_mean)/the_cov_prop*(ssy-the_mean)';
            target_prop = loglike_ind_prop + sum(log(exppdf(gamma_prop,reg_mean)));
            
            if (target_prop > target)
                gamma_curr = gamma_prop;
                loglike_ind_curr = loglike_ind_prop;
                break;
            end
            
            if (prop < curr)
                lower = prop;
            else
                upper = prop;
            end
            
        end
        
    end
    
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
    std_prop = std(ssx);
    the_cov = cov(ssx);
    the_cov = the_cov + diag((std_prop.*gamma_curr).^2);
    the_mean = mean(ssx);
    
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
        ssx_curr = ssx;
        std_curr = std_prop;
    end
    
    % store current values of the chain
    theta(i,:) = theta_curr;
    loglike(i) = loglike_ind_curr;   
    gamma(i,:) = gamma_curr;
    
end


end

