


%% real data model 2

load('radio_converted.mat');

NaN_Pos = isnan(Y);
n = 2000;
M = 50000;
lag = [1, 2, 4, 8];
ssy = summStat_quantiles(Y, lag);
cov_rw = [0.19,0.018,0.0034;0.018,0.011,0.0035;0.0034,0.0035,0.018];

start = [1.7, 35, 0.6];
simArgs = struct('ntoads',ntoads,'ndays',ndays,'model',2,'d0',NaN);
sumArgs = struct('lag',lag);
sumstat_fun = 'summStat_quantiles';

tic;
[theta,loglike] = bayes_sl_toads(ssy,n,M,cov_rw,start,simArgs,sumArgs,NaN_Pos,sumstat_fun);
time = toc;

%save('results_bsl_model2_realdata_n2000.mat', 'theta', 'loglike', 'time');


%% real data model 2 -- variance adjustment

load('radio_converted.mat');

NaN_Pos = isnan(Y);
n = 500;
M = 50000;
lag = [1, 2, 4, 8];
ssy = summStat_quantiles(Y, lag);
cov_rw = [0.3514    0.0453    0.0091;   0.0453    0.0228    0.0098;    0.0091    0.0098    0.0443];

start = [1.7, 35, 0.6];
simArgs = struct('ntoads',ntoads,'ndays',ndays,'model',2,'d0',NaN);
sumArgs = struct('lag',lag);
sumstat_fun = 'summStat_quantiles';
reg_mean = 0.5;

tic;
[theta,loglike,gamma] = bayes_sl_toads_variance(ssy,n,M,cov_rw,start,simArgs,sumArgs,NaN_Pos,reg_mean,sumstat_fun);
time = toc;

%save('results_bsl_model2_realdata_variance_n500.mat', 'theta', 'loglike', 'time', 'gamma');



%% real data model 2 -- mean adjustment

load('radio_converted.mat');

NaN_Pos = isnan(Y);
n = 500;
M = 50000;
lag = [1, 2, 4, 8];
ssy = summStat_quantiles(Y, lag);
cov_rw = [0.3055    0.0458    0.0139;    0.0458    0.0173    0.0072;    0.0139    0.0072    0.0351];

start = [1.7, 35, 0.6];
simArgs = struct('ntoads',ntoads,'ndays',ndays,'model',2,'d0',NaN);
sumArgs = struct('lag',lag);
sumstat_fun = 'summStat_quantiles';
tau = 0.5;

tic;
[theta,loglike,gamma] = bayes_sl_toads_mean(ssy,n,M,cov_rw,start,simArgs,sumArgs,NaN_Pos,tau,sumstat_fun);
time = toc;

%save('results_bsl_model2_realdata_mean_n500.mat', 'theta', 'loglike', 'time', 'gamma');





%% SIMULATED data model 2

load('data_toads_model2.mat');
[ndays,ntoads] = size(X);

NaN_Pos = isnan(X);
n = 300;
M = 50000;
lag = [1, 2, 4, 8];
ssy = summStat_quantiles(X, lag);
cov_rw = [0.1070    0.0073    0.0009;    0.0073    0.0018    0.0007;    0.0009    0.0007    0.0077];

start = theta_true;
simArgs = struct('ntoads',ntoads,'ndays',ndays,'model',2,'d0',NaN);
sumArgs = struct('lag',lag);
sumstat_fun = 'summStat_quantiles';

tic;
[theta,loglike] = bayes_sl_toads(ssy,n,M,cov_rw,start,simArgs,sumArgs,NaN_Pos,sumstat_fun);
time = toc;

%save('results_bsl_model2_simdata_n300.mat', 'theta', 'loglike', 'time');



%% SIMULATED data model 2 - variance adjustment

load('data_toads_model2.mat');
[ndays,ntoads] = size(X);

NaN_Pos = isnan(X);
n = 300;
M = 50000;
lag = [1, 2, 4, 8];
ssy = summStat_quantiles(X, lag);
cov_rw = [0.1070    0.0073    0.0009;    0.0073    0.0018    0.0007;    0.0009    0.0007    0.0077];

start = theta_true;
simArgs = struct('ntoads',ntoads,'ndays',ndays,'model',2,'d0',NaN);
sumArgs = struct('lag',lag);
sumstat_fun = 'summStat_quantiles';
reg_mean = 0.5;

tic;
[theta,loglike,gamma] = bayes_sl_toads_variance(ssy,n,M,cov_rw,start,simArgs,sumArgs,NaN_Pos,reg_mean,sumstat_fun);
time = toc;

%save('results_bsl_model2_simdata_variance_n300.mat', 'theta', 'loglike', 'gamma', 'time');


%% SIMULATED data model 2 - mean adjustment

load('data_toads_model2.mat');
[ndays,ntoads] = size(X);

NaN_Pos = isnan(X);
n = 300;
M = 50000;
lag = [1, 2, 4, 8];
ssy = summStat_quantiles(X, lag);
cov_rw = [0.0802    0.0054   -0.0015;    0.0054    0.0018    0.0009;   -0.0015    0.0009    0.0069];

start = theta_true;
simArgs = struct('ntoads',ntoads,'ndays',ndays,'model',2,'d0',NaN);
sumArgs = struct('lag',lag);
sumstat_fun = 'summStat_quantiles';
tau = 0.5;

tic;
[theta,loglike,gamma] = bayes_sl_toads_mean(ssy,n,M,cov_rw,start,simArgs,sumArgs,NaN_Pos,tau,sumstat_fun);
time = toc;

%save('results_bsl_model2_simdata_mean_n300.mat', 'theta', 'loglike', 'gamma', 'time');


