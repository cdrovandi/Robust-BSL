function [theta,loglike,gamma] = bayes_sl_misspec_mean(y,M,n,cov_rw,start,reg_mean,tau,samSize)

loglike=zeros(M,1);

theta_curr = start;
theta = zeros(M,1); %Initialising vector of estimates for motility and diffusivity
ssy1 = mean(y);
ssy2 = (var(y));
ssy = [ssy1 ssy2];

ns = length(ssy); %Total number of time periods including time zero
ssx = normrnd(theta_curr,1,n,samSize);
ssx = [mean(ssx,2) var(ssx,[],2)];

gamma_curr = reg_mean*[1,1];
gamma = zeros(M,ns);


%Calculating the mean and covariance of the summary statistics
std_curr = std(ssx);
the_mean = mean(ssx) + std_curr.*gamma_curr;
the_cov = cov(ssx);
ssx_curr = ssx;


L = chol(the_cov);
logdetA = 2*sum(log(diag(L)));

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
    
    
    theta_prop = mvnrnd(theta_curr,cov_rw); %Proposing a new pair of parameters
    
    %Simulating n data sets, finding summary statistics and then getting
    %the mean and covariance of these summary statistics
    ssx = normrnd(theta_prop,1,n,samSize);
    ssx = [mean(ssx,2) var(ssx,[],2)];
    
    %Calculating the mean and covariance of the summary statistics
    std_prop = std(ssx);
    the_cov = cov(ssx);
    the_mean = mean(ssx) + std_prop.*gamma_curr;
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
    gamma(i,:) = gamma_curr;
    
end


% end
