function [theta,loglike,epsilon] = bayes_sl_misspec(ssy,M,n,cov_rw,start,reg_mean)

loglike=zeros(M,1);

theta_curr = start;
theta = zeros(M,1); %Initialising vector of estimates

ns = length(ssy); %Total number of sample units
ssx = normrnd(theta_curr,1,n,50);
ssx = [mean(ssx,2) std(ssx,[],2)];

epsilon_curr = reg_mean*ones(1,ns);
epsilon = zeros(M,ns);

%Calculating the mean and covariance of the summary statistics
the_mean = mean(ssx);
the_cov = cov(ssx);
ssx_curr = ssx;
std_curr = std(ssx);
the_cov = the_cov + diag((std_curr.*epsilon_curr).^2);

L = chol(the_cov);
logdetA = 2*sum(log(diag(L)));

loglike_ind_curr = -0.5*logdetA - 0.5*(ssy-the_mean)/the_cov*(ssy-the_mean)';

for i = 1:M
    
    % update epsilon
    the_cov_base = cov(ssx_curr);
    the_mean = mean(ssx_curr);
    for j = 1:ns
        lower = 0; 
        target = loglike_ind_curr + sum(log(exppdf(epsilon_curr,reg_mean))) - exprnd(1);
        
        % step out for upper limit
        curr = epsilon_curr(j);
        upper = epsilon_curr(j) + 1;
        while(1)
            epsilon_upper = epsilon_curr;
            epsilon_upper(j) = upper;
            the_cov_upper = the_cov_base + diag((std_curr.*epsilon_upper).^2);
            L = chol(the_cov_upper);
            logdetA = 2*sum(log(diag(L)));
            loglike_ind_upper = -0.5*logdetA - 0.5*(ssy-the_mean)/the_cov_upper*(ssy-the_mean)';
            target_upper = loglike_ind_upper + sum(log(exppdf(epsilon_upper,reg_mean)));
            if (target_upper < target)
                break;
            end
            upper = upper + 1;
        end
        
        % shrink
        while(1)
            prop = unifrnd(lower,upper);
            epsilon_prop = epsilon_curr;
            epsilon_prop(j) = prop;
            
            the_cov_prop = the_cov_base + diag((std_curr.*epsilon_prop).^2);
            L = chol(the_cov_prop);
            logdetA = 2*sum(log(diag(L)));
            loglike_ind_prop = -0.5*logdetA - 0.5*(ssy-the_mean)/the_cov_prop*(ssy-the_mean)';
            target_prop = loglike_ind_prop + sum(log(exppdf(epsilon_prop,reg_mean)));
            
            if (target_prop > target)
                epsilon_curr = epsilon_prop;
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
    
    theta_prop = mvnrnd(theta_curr,cov_rw); %Proposing a new pair of parameters
    
    %Simulating n data sets, finding summary statistics and then getting
    %the mean and covariance of these summary statistics
    ssx = normrnd(theta_prop,1,n,50);
    ssx = [mean(ssx,2) std(ssx,[],2)];
    
    %Calculating the mean and covariance of the summary statistics
    std_prop = std(ssx);
    the_cov = cov(ssx);
    the_cov = the_cov + diag((std_prop.*epsilon_curr).^2);

    the_mean = mean(ssx);
    L = chol(the_cov);
    logdetA = 2*sum(log(diag(L)));
    
    loglike_ind_prop = -0.5*logdetA - 0.5*(ssy-the_mean)/the_cov*(ssy-the_mean)';
  
    % log prior densities to be sed within Metropolis step
    logprior_prop =  log(normlike([theta_prop sqrt(10)],1));
    logprior_curr = log(normlike([theta_curr sqrt(10)],1));
    
    % If the proposed parameters have a higher likelihood than previous, then
    % accept this new parameter at random.
     if (exp(loglike_ind_prop - loglike_ind_curr + logprior_curr-logprior_prop) > rand)
       %fprintf('*** accept ***\n');
        theta_curr = theta_prop;
        loglike_ind_curr = loglike_ind_prop;
        ssx_curr = ssx;
        std_curr = std_prop;
    end
    
    theta(i,:) = theta_curr;
    loglike(i) = loglike_ind_curr;
    epsilon(i,:) = epsilon_curr;
    
end


end

