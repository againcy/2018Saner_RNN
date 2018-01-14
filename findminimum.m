function [X, fX, i]=findminimum(f, X, maxIter)

alpha=0.00000007;
for i=1:maxIter
    [J, grad]=f(X);
    if norm(grad, 1)<1e-15
        break;
    end
    fprintf('Iterarion: %d | Cost: %f\n',i, J);
    fX(i)=J;
    X=X-alpha*grad;
end
