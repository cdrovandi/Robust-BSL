function theta = bayes_gibbs(y,M,start)
%M is the total number of steps attempted (iterations)
%n is number of simulated data sets
%cov_rw is the covariance matrix used in the random walk (use multivariate
%random normal for proposals)

theta = zeros(M,2); %Initialising vector of estimates for motility and diffusivity
theta(1,:) = start;

my = mean(y);
n = length(y);

for i = 2:M
    theta(i,1) = normrnd(my, sqrt(theta(i-1,2)/n));
    theta(i,2) = 1/gamrnd(n/2, 1/(0.5*sum((y - theta(i,1)).^2)));    
end

end

