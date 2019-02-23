function inverseDegreeMatrix(D)
    x,y = findn(D)
    for i = 1:length(x)
        if D[x[i], y[i]] != 0
            D[x[i], y[i]] = 1.0 / D[x[i],y[i]];
        end
    end
    return D
end