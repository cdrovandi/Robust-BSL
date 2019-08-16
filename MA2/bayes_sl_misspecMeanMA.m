function [theta,loglike,gamma] = bayes_sl_misspecMeanMA(ssy,M,n,start,reg_mean,y,tau)
% Robust BSL method with mean adjustment for MA2 example
%
% INPUT:
% ssy - observed data summaries 
% n - number of simulated data sets for estimating synthetic likelihood
% M - number of MCMC iterations
% cov_rw - the covariance matrix used in the random walk (uses multivariate random normal for proposals). Fixed in this example.
% start - initial value of parameters for Markov chain
% reg_mean - starting value for the adjustment procedure
% y - observed data
% tau - parameter of the prior for the gamma parameters of the mean adjustment
%
% OUTPUT:
% theta - MCMC samples of parameter values
% loglike - MCMC samples of estimated log likelihood
% gamma - MCMC samples of gamma parameter values of the mean adjustment
samSize=length(y);
loglike=zeros(M,1); %Storing mcmc values for log likelihoods

theta_curr = start; %Initial guesses for parameters
theta = zeros(M,2); %Storing mcmc chain for parameters

ns = length(ssy); %Total number of summaries

ssx = zeros(n,ns);

% simulating n data sets
nu = randn(samSize,n);

% initial simulated data
x(1,:) = nu(1,:);
x(2,:) = nu(2,:)+theta_curr(1)*nu(1,:);

for t=3:samSize
    x(t,:) = nu(t,:)+theta_curr(1)*nu(t-1, :)+theta_curr(2)*nu(t-2,:);
end
% summary statistics functions
  s0=@(x) mean((x-mean(x)).^2);
  s1=@(x) mean(x(1:(samSize-1),:).*x(2:samSize,:));
  s2=@(x) mean(x(1:(samSize-2),:).*x(3:samSize,:));
  s3=@(x) mean(x(1:(samSize-3),:).*x(4:samSize,:));

% Generating simulated summary statistics
  
  ssx = [s0(x)' s1(x)' s2(x)'];
  
gamma_curr = reg_mean*ones(1,ns); % initial value of gamma mean adjustment parameters
gamma = zeros(M,ns);% store mcmc chain for gamma parameters

%Calculating the mean and covariance of the summary statistics
ssx_curr = ssx;
std_curr = std(ssx);
the_mean = mean(ssx) + std_curr.*gamma_curr;
the_cov = cov(ssx);

% estimate logdet numerically stably
L = chol(the_cov);
logdetA = 2*sum(log(diag(L)));

% synthetic likelihood
loglike_ind_curr = -0.5*logdetA - 0.5*(ssy-the_mean)/the_cov*(ssy-the_mean)';

for i = 1:M
    i
    % update gamma
    the_cov = cov(ssx_curr);
     L = chol(the_cov);
    logdetA = 2*sum(log(diag(L)));
    the_mean_base = mean(ssx_curr);
    
    for j = 1:ns
        target = loglike_ind_curr + sum(-log(tau) - abs(gamma_curr)/tau) - exprnd(1);
        
        curr = gamma_curr(j);
        lower = curr - 1;
        upper = curr + 1;

        % step out for lower limit
       while(1)
            gamma_lower = gamma_curr;
            gamma_lower(j) = lower;
            the_mean_lower = the_mean_base + std_curr.*gamma_lower;
            loglike_ind_lower = -0.5*logdetA - 0.5*(ssy-the_mean_lower)/the_cov*(ssy-the_mean_lower)';
            target_lower = loglike_ind_lower + sum(-log(tau) - abs(gamma_lower)/tau);
            if (target_lower < target)
                break;
            end
            lower = lower - 1;
       end
       
       % step out for upper limit
        while(1)
            gamma_upper = gamma_curr;
            gamma_upper(j) = lower;
            the_mean_upper = the_mean_base + std_curr.*gamma_upper;
            loglike_ind_upper = -0.5*logdetA - 0.5*(ssy-the_mean_upper)/the_cov*(ssy-the_mean_upper)';
            target_upper = loglike_ind_upper + sum(-log(tau) - abs(gamma_upper)/tau);
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
            
            the_mean_prop = the_mean_base + std_curr.*gamma_prop;
            loglike_ind_prop = -0.5*logdetA - 0.5*(ssy-the_mean_prop)/the_cov*(ssy-the_mean_prop)';
            target_prop = loglike_ind_prop + sum(-log(tau) - abs(gamma_prop)/tau);
            
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
    
t_all = 0;
while (t_all<3)    
   
    theta_prop =mvnrnd(theta_curr,[.01 -.0;-.0 .01]); %Proposing a new pair of parameters

    % Rejecting proposed draws if they are outside the parameter space
    t1 = (abs(theta_prop(1))<1);
    t2 = (theta_prop(1)+theta_prop(2)>-1);
    t3 = (theta_prop(1)-theta_prop(2)<1);
    
 t_all = t1+t2+t3;
end 
    %simulating n data sets using the proposed parameters
    nu = randn(samSize,n);

    x(1,:) = nu(1,:);
    x(2,:) = nu(2,:)+theta_prop(1)*nu(1,:);
  for t=3:samSize
    x(t,:) = nu(t,:)+theta_prop(1)*nu(t-1, :)+theta_prop(2)*nu(t-2,:);
  end
  ssx = [s0(x)' s1(x)' s2(x)'];
  
    %Calculating the mean and covariance of the summary statistics
    std_prop = std(ssx);
    the_cov = cov(ssx);
    the_mean = mean(ssx) + std_prop.*gamma_curr;
  
    % estimate logdet numerically stably
    L = chol(the_cov);
    logdetA = 2*sum(log(diag(L)));
    
    % synthetic likelihood
    loglike_ind_prop = -0.5*logdetA - 0.5*(ssy-the_mean)/the_cov*(ssy-the_mean)';
    
    % Metropolis-Hastings accept/reject
    if (exp(loglike_ind_prop - loglike_ind_curr) > rand)
        %fprintf('*** accept ***\n');
        theta_curr = theta_prop;
        loglike_ind_curr = loglike_ind_prop;
        ssx_curr = ssx;
        std_curr = std_prop;
    end
 i   
    % store current values of the chain
    theta(i,:) = theta_curr;
    loglike(i) = loglike_ind_curr;
    gamma(i,:) = gamma_curr;
    
end


end

