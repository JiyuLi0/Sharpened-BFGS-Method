function res = loss_gradient(X, Y, w, mu)
    temp = size(Y);
    N = temp(1);
    h = exp(-Y.*(X*w));
    res = 1./N*sum(-Y.*h./(1 + h).*X)' + mu*w;
end
%y