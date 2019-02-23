module DiffusionRunner
export runWithCommGroundTruth

include("utils.jl")

include("nnlinear_diffusion.jl")

# This code is converted from Heat Kernel Diffusion paper MATLAB Code.
function runWithCommGroundTruth(dataFilename::AbstractString, commFilename::AbstractString, h::Float64, t::Float64, p::Float64, truncCutoff::Float64)

# Load the provided graph.

A = DiffusionUtils.LoadGraph(dataFilename)
C = DiffusionUtils.LoadCommunities(commFilename)

n = size(A, 1)
d = sum(A, 2)
x = find(d)
pinvD = sparse(x, x, 1.0./d[x])
L = speye(n,n) - A * pinvD
totalcommunities = 100
bestfmeas = zeros(totalcommunities, 1)
bestrecsize = zeros(totalcommunities, 1)
condofbestfmeas = zeros(totalcommunities, 1)
# find the first community with size > 10
e = ones(Int, n, 1)
commsize = sum(C, 1)
comm1 = minimum(find(commsize .> 10))

# check every 10th community after the first one
# that has size > 10
testcomms = zeros(Int, totalcommunities, 1)
for i=1:totalcommunities
    testcomms[i] = comm1 + 10*(i-1)
end

println("running...")
for numcom = 1:totalcommunities
	@printf("Community number %i out of 100.\n", numcom)
    comm = testcomms[numcom]
    verts = find(C[:,comm])
    deg = length(verts)
    recalls = zeros(deg, 1)
    precisions = zeros(deg, 1)
    fmeas = zeros(deg, 1)
    conds = zeros(deg, 1)
    for trial = 1:deg
        bset, conds[trial, 1] = NonlinearDiffusion.run_nnlinear_diffusion(A, L, pinvD, h, verts[trial], t, p, truncCutoff)
        recalls[trial,1] = length(intersect(verts,bset))/length(verts)
        precisions[trial,1] = length(intersect(verts,bset))/length(bset)
        fmeas[trial,1] = 2 * recalls[trial,1] * precisions[trial,1] /(recalls[trial,1]+precisions[trial,1])
		if fmeas[trial,1] > bestfmeas[numcom,1]
            bestfmeas[numcom,1] = fmeas[trial,1]
            bestrecsize[numcom,1] = length(bset)
            condofbestfmeas[numcom,1] = conds[trial,1]
        end
    end
end
@printf("mean fmeas=%6.4f \t mean setsize=%6.4f \t mean cond=%6.4f", sum(bestfmeas[:,1])/totalcommunities, sum(bestrecsize[:,1])/totalcommunities, sum(condofbestfmeas[:,1])/totalcommunities)
end
end