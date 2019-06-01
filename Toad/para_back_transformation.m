function [theta] = para_back_transformation(theta_tilde)

% computes back transformation of parameter for toad example
% INPUT:
% theta_tilde - parameter on the transformed space
% OUTPUT:
% theta - parameter on original space

e_theta_tilde = exp(theta_tilde);

a = [1, 0, 0]; % lower bounds
b = [2, 100, 0.9]; % upper bounds
% back transform
theta = a ./ (1 + e_theta_tilde) + b ./ (1 + 1 ./ e_theta_tilde);

end

