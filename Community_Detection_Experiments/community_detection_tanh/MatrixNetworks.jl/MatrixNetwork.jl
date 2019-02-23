# This code is taken from https://github.com/dgleich/MatrixNetworks.jl

# create type MatrixNetwork
type MatrixNetwork{T}
    n::Int64 # number of columns/rows
    rp::Vector{Int64} # row pointers
    ci::Vector{Int64} # column indices
    vals::Vector{T} # corresponding values
end

function MatrixNetwork{T}(A::SparseMatrixCSC{T,Int64})
    At = A'
    return MatrixNetwork(size(At,2),At.colptr,At.rowval,At.nzval)
end

function MatrixNetwork(ei::Vector{Int64},ej::Vector{Int64})
    At = sparse(ej,ei,true);
    return MatrixNetwork(size(At,2),At.colptr,At.rowval,At.nzval)
end


import Base.sparse, Base.size

"""
Return back an adjacency matrix representation
of the transpose. This requires no work. 
"""
function sparse_transpose{T}(A::MatrixNetwork{T})
    return SparseMatrixCSC(A.n,A.n,A.rp,A.ci,A.vals)
end

"""
Return back an adjacency matrix representation
of the current MatrixNetwork
"""
function sparse{T}(A::MatrixNetwork{T})
    return sparse_transpose(A)'
end

function size(A::MatrixNetwork)
    return (A.n,A.n)
end

function size(A::MatrixNetwork, dim::Integer)
    if dim == 1 || dim == 2
        return A.n
    elseif dim > 2
        return 1
    else
        throw(DomainError())
    end
end