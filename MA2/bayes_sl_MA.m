  function [theta,loglike] = bayes_sl_MA(ssy,M,n,start,y)
% BSL method for MA2 example
% INPUT:
% ssy - observed data summaries 
% n - number of simulated data sets for estimating synthetic likelihood
% M - number of MCMC iterations
% cov_rw - the covariance matrix used in the random walk (uses multivariate random normal for proposals). Fixed in this example.
% start - initial value of parameters for Markov chain
% reg_mean - starting value for the adjustment procedure (and hyperparameter for variance adjustment)
% y - observed data
%
% OUTPUT:
% theta - MCMC samples of parameter values
% loglike - MCMC samples of estimated log likelihood
% gamma - MCMC samples of gamma parameter values of the mean adjustment

samSize=length(y);
loglike=zeros(M,1);

theta_curr = start;%Initial guesses for parameters
theta = zeros(M,2); %Storing mcmc chain for parameters

ns = length(ssy); %Total number of summary statistics 

ssx = zeros(n,ns);%Initial summary statistics

% simulating n data sets
nu = randn(samSize+100,n);

% initial simulated data
x(1,:) = nu(1,:);
x(2,:) = nu(2,:)+theta_curr(1)*nu(1,:);


for t=3:samSize+100
    x(t,:) = nu(t,:)+theta_curr(1)*nu(t-1, :)+theta_curr(2)*nu(t-2,:);
end

  x=x(100:samSize+100,:);

% summary statistics functions
  s0=@(x) mean((x-mean(x)).^2);
  s1=@(x) mean(x(1:(samSize-1),:).*x(2:samSize,:));
  s2=@(x) mean(x(1:(samSize-2),:).*x(3:samSize,:));
  s3=@(x) mean(x(1:(samSize-3),:).*x(4:samSize,:));
  
  
     ssx = [s0(x)' s1(x)' s2(x)'];


%Calculating the mean and covariance of the summary statistics
the_mean = mean(ssx);
the_cov = (cov(ssx));

 
% estimate logdet numerically stably
L = chol(the_cov);
logdetA = 2*sum(log(diag(L)));

% synthetic likelihood
loglike_ind_curr = - 0.5*(ssy-the_mean)/the_cov*(ssy-the_mean)';

for i = 1:M

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
nu = randn(samSize+100,n);

x(1,:) = nu(1,:);
x(2,:) = nu(2,:)+theta_prop(1)*nu(1,:);


for t=3:samSize+100
    x(t,:) = nu(t,:)+theta_prop(1)*nu(t-1, :)+theta_prop(2)*nu(t-2,:);
end
 x=x(100:samSize+100,:);

    ssx = [s0(x)' s1(x)' s2(x)'];
  
%Calculating the mean and covariance of the summary statistics
    
    the_mean = mean(ssx);
    the_cov = cov(ssx);

 % estimate logdet numerically stably    
    L = chol(the_cov);
    logdetA = 2*sum(log(diag(L)));
    
% Synthetic likelihood 
loglike_ind_prop = - 0.5*(ssy-the_mean)/the_cov*(ssy-the_mean)';

% Metropolis-Hastings accept/reject
    if (exp(loglike_ind_prop - loglike_ind_curr) > rand)
        %fprintf('*** accept ***\n');
        theta_curr = theta_prop;
        loglike_ind_curr = loglike_ind_prop;    
    end
 i   
% store current values of the chain
    theta(i,:) = theta_curr;
    loglike(i) = loglike_ind_curr;

    
end


  end

