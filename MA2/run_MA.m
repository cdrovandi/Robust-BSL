clear
clc
%% MA2 BSL example
rng(1235)
load('BSL_MA2_data.mat')  

%%%%%%%%%%%%%%% True Values for simulated SV process %%%%%%%%%%%%%%%
% T=100; Data size
% M=50000; MCMC iterations
% n=10; Simulated data sets for BSL
%
% omgv = -.76; Constant for log-AR(1) SV model
% rho = .9; Persistence parameter
% sigv = .36; Standard deviation for SV model
% 
%  theta_true = [omgv,rho,sigv]';
%  y= sv_make(theta_true,2*T); Generating data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summary statistics used for this example
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s0=@(x) mean((x-mean(x)).^2);
s1=@(x) mean(x(1:(T-1),:).*x(2:T,:));
s2=@(x) mean(x(1:(T-2),:).*x(3:T,:));
s3=@(x) mean(x(1:(T-3),:).*x(4:T,:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vector summary statistic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ssy = [s0(y) s1(y)' s2(y)'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Starting values for the chain obtained from MA(2) model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[parameters, errors, LLF , SEregression,stderrors, robustSE,   scores, likelihoods]=armaxfilter(y,1,0,2);

% Starting values for the MCMC chain
start = parameters(2:end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCMC Implementation for the R-BSL-V example for the MA(2) model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Starting value for the adjustment approach 
reg_mean = .3;

[thetaRV,loglike,epsilon]=bayes_sl_misspecMA(ssy,M,n,start,reg_mean,y);
% Acceptance rate for R-BSL-V
acc_rate_RbslV = mean(diff(thetaRV(:,1))~=0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCMC Implementation for the R-BSL-M example for the MA(2) model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hyperparameter for the Laplace prior.  
tau=.5;

[thetaRM,loglike,gamma]=bayes_sl_misspecMeanMA(ssy,M,n,start,reg_mean,y,tau);
% Acceptance rate for R-BSL-V
acc_rate_RbslM = mean(diff(thetaRM(:,1))~=0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCMC Implementation for the R-BSL-M example for the MA(2) model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[theta,loglike]=bayes_sl_MA(ssy,M,n,start,y);
% Acceptance rate for BSL
acc_rate_bsl = mean(diff(theta(:,1))~=0);


%% Thining the samples by taking every tenth observations %%

Burn = 10000; % Burn-in period for the MCMC sampler
thin = 10; % Thinning rate for the chain


ThetaV = thetaRV(Burn:end,:);
seq = 1:thin:length(ThetaV);
% Thinned R-BSL-V draws
Thin_ThetaV = ThetaV(seq,:);
% Acceptance rate for thinned R-BSL-V draws
Thin_acc_rate_RbslV= mean(diff(Thin_ThetaV(:,1))~=0);

ThetaM = thetaRM(Burn:end,:);
% Thinned R-BSL-M draws
Thin_ThetaM= ThetaM(seq,:);
% Acceptance rate for thinned R-BSL-M draws
Thin_acc_rate_RbslM= mean(diff(Thin_ThetaM(:,1))~=0);

Theta = theta(Burn:end,:);
% Thinned BSL draws
Thin_Theta = Theta(seq,:);
% Acceptance rate for thinned BSL draws
Thin_acc_rate_bsl= mean(diff(Thin_Theta(:,1))~=0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kernel densities for the different parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[f1,x1]=ksdensity(Thin_Theta(:,1));
[f2,x2]=ksdensity(Thin_Theta(:,2));
[fm1,xm1]=ksdensity(Thin_ThetaM(:,1));
[fm2,xm2]=ksdensity(Thin_ThetaM(:,2));
[fv1,xv1]=ksdensity(Thin_ThetaV(:,1));
[fv2,xv2]=ksdensity(Thin_ThetaV(:,2));

%% Plotting The Thetas 
subplot(1,2,1)
plot(x1,f1,'blue',xv1,fv1,'black',xm1,fm1,'green')
subplot(1,2,2)
plot(x2,f2,'blue',xv2,fv2,'black',xm2,fm2,'green')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing kernel densities for the different adjustment terms 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[g1,xg1]=ksdensity(gamma(seq,1));
[g2,xg2]=ksdensity(gamma(seq,2));
[g3,xg3]=ksdensity(gamma(seq,3));
[e1,xe1]=ksdensity(epsilon(seq,1),'support','positive');
[e2,xe2]=ksdensity(epsilon(seq,2),'support','positive');
[e3,xe3]=ksdensity(epsilon(seq,3),'support','positive');
%% Plotting The Adjustments terms  
subplot(2,3,1)
plot(xg1,g1)
subplot(2,3,2)
plot(xg2,g2)
subplot(2,3,3)
plot(xg3,g3)
subplot(2,3,4)
plot(xe1,e1)
subplot(2,3,5)
plot(xe2,e2)
subplot(2,3,6)
plot(xe3,e3)






