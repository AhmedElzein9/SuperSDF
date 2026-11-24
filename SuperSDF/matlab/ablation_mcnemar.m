function [p, b, c] = ablation_mcnemar(y_true, y_a, y_b)
%ABLATION_MCNEMAR Exact two-sided McNemar test between two classifiers.
%   [p, b, c] = ablation_mcnemar(y_true, y_a, y_b)
%   b = #samples A correct and B wrong, c = #samples A wrong and B correct.
%   p = min(1, 2*binocdf(min(b,c), b+c, 0.5)); p = 1 when b+c = 0.
ca = y_a(:) == y_true(:);
cb = y_b(:) == y_true(:);
b = sum(ca & ~cb);
c = sum(~ca & cb);
n = b + c;
if n == 0
    p = 1;
else
    p = min(1, 2 * binocdf(min(b,c), n, 0.5));
end
end
