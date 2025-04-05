function res = loss_hessian(X, Y, w, mu)
    [N, d] = size(X);
    h = exp(-Y.*(X*w));
    I = eye(d);
    res = 1./N*X'*(h.*Y.*Y./((1 + h).*(1 + h)).*X) + mu*I;
end
%y