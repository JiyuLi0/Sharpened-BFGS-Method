function B = normalize_row(A)
    [m, d] = size(A);
    B = zeros(m ,d);
    for i = 1:m
        row_norm = norm(A(i, :));
        if row_norm == 0
            B(i, :) = A(i, :);
        else
            B(i, :) = A(i, :)./norm(A(i, :));
        end
    end
end