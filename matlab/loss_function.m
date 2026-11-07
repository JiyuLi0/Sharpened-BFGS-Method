function res = loss_function(X, Y, w, mu)
    temp = size(Y);
    N = temp(1);
    res = 1./N*sum(log(1 + exp(-Y.*(X*w)))) + mu*(w'*w)./2;
end
%y