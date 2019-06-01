function s = summStat_quantiles(X,lag)
% Summary statistics for the toad example for all lags inputted
%
% INPUT:
% X - toads moves data matrix, dimension is ndays by ntoads
% lag - lags to compute the summary statsitics for
% OUTPUT:
% ssx - summary statistics for all lags inputted

nlag = length(lag);

x = cell(nlag,1);
s = [];  
for k = 1:nlag
    l = lag(k);
    x = obsMat2deltax(X, l);
    x = abs(x);
    
    return_ind = x<10;
    x_noret = x(~return_ind);
    
    s = [s sum(return_ind) log(diff(quantile(x_noret, 0:0.1:1))) median(x_noret)];
end

end