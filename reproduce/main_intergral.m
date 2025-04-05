%%%%%%%%%%%%%%%%%%%%%%%%%% Model Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%

[Y, X] = libsvmread('gisette_scale');
X(isnan(X)) = 0;
X_dense = full(X);
X = normalize_row(X);
[N, d] = size(X);
X_new = zeros(size(X,1), d);
Y_new = zeros(size(X,1), 1);
iter = 1;
iternumber=8000;
for k = 1:N
if Y(k) == 0
X_new(iter, :) = X(k, :);
Y_new(iter) = 1;
iter = iter + 1;
end
if Y(k) == 2
X_new(iter, :) = X(k, :);
Y_new(iter) = -1;
iter = iter + 1;
end
if Y(k) == 1
X_new(iter, :) = X(k, :);
Y_new(iter) = 1;
iter = iter + 1;
end
if Y(k) == -1
X_new(iter, :) = X(k, :);
Y_new(iter) = -1;
iter = iter + 1;
end
end
X=X_new;
Y=Y_new;
w_0 = ones(d, 1)./(d*sqrt(d));
mu = 0.00001;
L=0.2;
%L = mu + 0.25;
g_0 = loss_gradient(X, Y, w_0, mu);
H_0 = loss_hessian(X, Y, w_0, mu);
lambda_0 = sqrt(g_0'*(H_0\g_0));

disp("Model Initialization Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%% Gradient Descent %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_gd = [1];
w = w_0;

for iter = 1:iternumber
    w = w - 1./L*loss_gradient(X, Y, w, mu);
    g = loss_gradient(X, Y, w, mu);
    H = loss_hessian(X, Y, w, mu);
    lambda = sqrt(g'*(H\g));
    y_gd = [y_gd, lambda/lambda_0];
end

disp("Gradient Descent Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BFGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_bfgs = [1];
w = w_0;
B = L*eye(d);

for iter = 1:iternumber
    w_new = w - B \ loss_gradient(X, Y, w, mu);
    s = w_new - w;
    %y = loss_gradient(X, Y, w_new, mu) - loss_gradient(X, Y, w, mu);
    H = loss_hessian(X, Y, w, mu);
    H_new = loss_hessian(X, Y, w_new, mu);
    H_avg = 0.5 * (H + H_new);
    y = H_avg * s;
    a = B*s;
    b = s'*a;
    B = B - (a*a')/b + (y*y')/(s'*y);
    w = w_new;
    g = loss_gradient(X, Y, w, mu);
    H = loss_hessian(X, Y, w, mu);
    lambda = sqrt(g'*(H\g));
    y_bfgs = [y_bfgs, lambda/lambda_0];
end

disp("BFGS Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%% Greedy BFGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_greedy = [1];
w = w_0;
B = L*eye(d);
M = 0;

for iter = 1:iternumber
    w_new = w - B \ loss_gradient(X, Y, w, mu);
    s = w_new - w;
    %y = loss_gradient(X, Y, w_new, mu) - loss_gradient(X, Y, w, mu);
    H = loss_hessian(X, Y, w, mu);
    H_new = loss_hessian(X, Y, w_new, mu);
    H_avg = 0.5 * (H + H_new);
    y = H_avg * s;
    r = sqrt(s'*H*s);
    B_bar = (1 + M*r)*B;
    index = 1;
    max_res = B_bar(1, 1)/H_new(1, 1);
    for i = 2:d
        temp = B_bar(i, i)/H_new(i, i);
        if temp > max_res
            max_res = temp;
            index = i;
        end
    end
    a = B_bar(:, index);
    b = H_new(:, index);
    B = B_bar - (a*a')/B_bar(index, index) + (b*b')/H_new(index, index);
    w = w_new;
    g = loss_gradient(X, Y, w, mu);
    H = loss_hessian(X, Y, w, mu);
    lambda = sqrt(g'*(H\g));
    y_greedy = [y_greedy, lambda/lambda_0];
end

disp("Greedy BFGS Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%% Sharpened BFGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_Sharpened = [1];
w = w_0;
B = L*eye(d);
M = 0;

for iter = 1:iternumber
    w_new = w - B \ loss_gradient(X, Y, w, mu);
    s = w_new - w;
    %y = loss_gradient(X, Y, w_new, mu) - loss_gradient(X, Y, w, mu);
    H = loss_hessian(X, Y, w, mu);
    H_new = loss_hessian(X, Y, w_new, mu);
    H_avg = 0.5 * (H + H_new);
    y = H_avg * s;
    a = B*s;
    b = s'*a;
    B = B - (a*a')/b + (y*y')/(s'*y);
    H = loss_hessian(X, Y, w, mu);
    H_new = loss_hessian(X, Y, w_new, mu);
    r = sqrt(s'*H*s);
    B_bar = (1 + 0.5*M*r)*(1 + 0.5*M*r)*B;
    index = 1;
    max_res = B_bar(1, 1)./H_new(1, 1);
    for i = 2:d
        temp = B_bar(i, i)./H_new(i, i);
        if temp > max_res
            max_res = temp;
            index = i;
        end
    end
    a = B_bar(:, index);
    b = H_new(:, index);
    B = B_bar - (a*a')/B_bar(index, index) + (b*b')/H_new(index, index);
    w = w_new;
    g = loss_gradient(X, Y, w, mu);
    H = loss_hessian(X, Y, w, mu);
    lambda = sqrt(g'*(H\g));
    y_Sharpened = [y_Sharpened, lambda/lambda_0];
end

disp("Sharpened BFGS Finish");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = 0:iternumber;
semilogy(x, y_bfgs,'-.', 'Color', '#0000FF', 'LineWidth', 2);

hold on
semilogy(x, y_gd, '-.', 'Color', '#000000', 'LineWidth', 2);
semilogy(x, y_greedy, '-.', 'Color', '#FFA500', 'LineWidth', 2);
semilogy(x, y_Sharpened, '-.', 'Color', '#FF5733', 'LineWidth', 2);

l = legend({'BFGS', 'GD', ' Greedy', 'Sharpened'});
set(l, 'Interpreter', 'latex', 'fontsize', 15, 'Location', 'southwest')
xlabel('Number of iterations $t$','Interpreter','latex', 'fontsize', 20);
ylabel('$\frac{\lambda_{f}(x_t)}{\lambda_{f}(x_0)}$', 'Interpreter', 'latex', 'fontsize', 20, 'Rotation', 0);
xlim([0 iternumber]);
ylim([1e-15 1e0]);
ax = gca;
ax.FontSize = 15;
set(gcf,'position',[0,0,600,400])
hold off