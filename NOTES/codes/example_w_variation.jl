
function main()
    true_w = [-2.0, 3.0]
    Nsizes = range(20, stop=1000, step=20)
    Nsample = 10000; # Number of datasets
    all_ss = zeros(length(Nsizes),Nsample)
    for j in 1:length(Nsizes)
        N = Nsizes[j]   # Number of objects        
        x = rand(N)     # random data
        # Build matrix X
        X = zeros(N,2)
        for i in 1:N
            X[i,1] = 1.0
            X[i,2] = x[i]
        end
        noisevar = 0.5^2
        for i in 1:Nsample
            t = X*true_w + randn(N)*sqrt(noisevar)
            w = inv(X'*X) * X' * t
            σ2 = (t'*t - t'*X*w)/Ndata
            all_ss[j,i] = σ2
        end
    end
end