function [H] = MKC_approximation(KH, index, numclass, alpha)

%%%% Input: 1. KH (n*n*m) consists of m base kernel matrices, where n is the
%%%%        sample number and m is the number of kernels.
%%%%        2. index is the index set of selected landmarks.
%%%%        3. numclass is the number of clusters.
%%%%        4. alpha is the kernel weights learned by some multiple kernel
%%%%        clustering algorithm on landmarks.

%%%% Output: H is the approximation embedding of the whole training datasets

numker = size(KH, 3);

if numker == 1
    KH_small = KH(index,index);
    [H_small, eigvalue] = eigs(KH_small, numclass, 'LA');
    P = KH(:,index);
    H = P * H_small / eigvalue;
    
elseif numker > 1
    P = KH(:,index,:);
    KH_small = KH(index,index,:);
    KH_alpha = mycombFun(KH_small, alpha.^2);
    P_alpha = mycombFun(P, alpha.^2);
    [H_small, eigvalue] = eigs(KH_alpha, numclass, 'LA');
    H = P_alpha * H_small / eigvalue;
    
end

end

function cF = mycombFun(Y,gamma)

m = size(Y,3);
n = size(Y,1);
s = size(Y,2);
cF = zeros(n,s);
for p =1:m
    cF = cF + Y(:,:,p)*gamma(p);
end
end