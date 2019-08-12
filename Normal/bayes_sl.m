function [theta, loglike] = bayes_sl(ssy,M,n,cov_rw,start)
%M is the total number of steps attempted (iterations)
%n is number of simulated data sets
%cov_rw is the covariance matrix used in the random walk (use multivariate
%random normal for proposals)

loglike=zeros(M,1);

theta_curr = start;
theta = zeros(M,1); %Initialising vector of estimates for motility and diffusivity

ns = length(ssy); %Total number of time periods including time zero
ssx = normrnd(theta_curr,1,n,50);
ssx = [mean(ssx,2) std(ssx,[],2)];

%Calculating the mean and covariance of the summary statistics
the_mean = mean(ssx);
the_cov = cov(ssx);
L = chol(the_cov);
logdetA = 2*sum(log(diag(L)));

%Calculating the (log) likelihood of getting the same summary statistics as
%the data if the mean of the summary statistics is the_mean and the
%covariance of the summary statistics is the_cov, OR a measure of probability that
%the simulated data come from the same distribution as the observed data
loglike_ind_curr = -0.5*logdetA - 0.5*(ssy-the_mean)*inv(the_cov)*(ssy-the_mean)';

for i = 1:M

    theta_prop = mvnrnd(theta_curr,cov_rw); %Proposing a new pair of parameters
    
    %Simulating n data sets, finding summary statistics and then getting
    %the mean and covariance of these summary statistics
    ssx = normrnd(theta_prop,1,n,50);
    ssx = [mean(ssx,2) std(ssx,[],2)];
    
    the_mean = mean(ssx);
    the_cov = cov(ssx);
    L = chol(the_cov);
    logdetA = 2*sum(log(diag(L)));
    
    %Finding the log likelihood
    %loglike_ind_curr
    loglike_ind_prop = -0.5*logdetA - 0.5*(ssy-the_mean)*inv(the_cov)*(ssy-the_mean)';
    % log prior densities to be sed within Metropolis step
    logprior_prop =  log(normlike([theta_prop sqrt(10)],1));
    logprior_curr = log(normlike([theta_curr sqrt(10)],1));
    
    % If the proposed parameters have a higher likelihood than previous, then
    % accept this new parameter at random.
     if (exp(loglike_ind_prop - loglike_ind_curr + logprior_curr-logprior_prop) > rand)
        %fprintf('*** accept ***\n');
        theta_curr = theta_prop;
        loglike_ind_curr = loglike_ind_prop;
    end
    theta(i,:) = theta_curr;
    loglike(i)=loglike_ind_curr;   
end
end

