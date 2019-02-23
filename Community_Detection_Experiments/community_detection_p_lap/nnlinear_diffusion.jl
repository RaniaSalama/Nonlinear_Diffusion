module NonlinearDiffusion
export run_nnlinear_diffusion
include("MatrixNetworks.jl\\MatrixNetworks.jl")


function run_nnlinear_diffusion{T}(A::SparseMatrixCSC{T,Int64}, N::SparseMatrixCSC{T,Int64}, pinvD::SparseMatrixCSC{T,Int64}, h::Float64, seed::Int64, t::Float64, p::Float64, truncCutoff::Float64)
    n = size(A, 1)
    u = sparsevec([seed], [1], n, +)
	tt = 0
    while tt < t
	    diff_u = N * pinvD * u;
        u = u - h * N' * ((abs(diff_u)).^(p-1) .* sign(diff_u)) 
		u[u .<  truncCutoff] = 0.
		u[u .> 1] = 1.
		tt = tt + 1
    end 
	sweepProfile = MatrixNetworks.sweepcut(A, full(u))
	bsetind = indmin(sweepProfile.conductance[1:sweepProfile.number_of_nnz_nodes])
	set = MatrixNetworks.bestset(sweepProfile)
	return Set(set), sweepProfile.conductance[bsetind]
end

end