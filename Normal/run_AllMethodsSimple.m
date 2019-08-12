

% clear
% clc
sigmas = 0.2:0.1:2; % Grid for sigmas
M=10000; % Number of iterations for each MC
n = 100; % Number of synthetic data sets
start = 1.0; % Starting values
reg_mean = 0.3; % initial starting point for Gammas

%% Loading synthetic data for experiments: Fixed throughout
load('data.mat'); 
%% Loop for the different methods over the sigma grid 
for i = 1:length(sigmas)
    
   samSize=length(r);
   sigma_true = sigmas(i)
   y = 1 + sigma_true*r;
    
    theta = bayes_gibbs(y,M,[1 sigma_true^2]);
    
    cov_rw1(i) = var(theta(:,1));
    mu_true(i) = mean(theta(:,1));
    
    cov_rw=cov_rw1(i);
    
    ssy = [mean(y) std(y)];
    [theta, loglike] = bayes_sl(ssy,M,n,cov_rw,start);
    
    mu_bsl(i) = mean(theta(:,1));
    acc_rate_bsl(i) = mean(diff(theta(:,1))~=0);
    
%%%%% Variance Incompatibility approach  %%%%%
    [theta, loglike, gamma] = bayes_sl_misspec(ssy,M,n,cov_rw,start,reg_mean);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Zeta_bsl_rob(:,:,i) = [theta(:,1) gamma];
    acc_rate_bsl_rob(i) = mean(diff(theta(:,1))~=0);
    
%%%%% Mean Incompatibility: LAPLACE prior w/tau=0.5  %%%%%
  [theta_mean1,t1,gamma_1] = bayes_sl_misspec_mean(y,M,n,cov_rw,start,reg_mean,.5,samSize); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Zeta_RBSL1(:,:,i) = [theta_mean1 gamma_1];
    acc_rate_Rbsl1(i) = mean(diff(theta_mean1)~=0);

    
end



%% Plotting Acceptance rates
plot(sigmas, acc_rate_bsl,'red',sigmas, acc_rate_Rbsl1,'green',sigmas, acc_rate_bsl_rob,'black')
% 
% 
Theta_BSL_robM1 = squeeze(Zeta_RBSL1(:,1,:));
Theta_BSL_rob = squeeze(Zeta_bsl_rob(:,1,:));
Gammas_M = squeeze(Zeta_RBSL1(:,2:3,:));
Gammas_V = squeeze(Zeta_bsl_rob(:,2:3,:));
% 
% 
% 
% 
subplot(2,2,1)
plot(sigmas,median(Theta_BSL_rob),sigmas,quantile(Theta_BSL_rob,.025,1),'red',sigmas,quantile(Theta_BSL_rob,.975),'red')

subplot(2,2,3)
plot(sigmas,median(Theta_BSL_robM1),sigmas,quantile(Theta_BSL_robM1,.025,1),'red',sigmas,quantile(Theta_BSL_robM1,.975,1),'red')

subplot(2,2,[2,4])
plot(sigmas, acc_rate_bsl,'red',sigmas, acc_rate_Rbsl1,'green',sigmas, acc_rate_bsl_rob,'black')
