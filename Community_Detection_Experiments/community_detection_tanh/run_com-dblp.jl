include("run_against_ground_truth.jl")
using DiffusionRunner
runner = DiffusionRunner

t = 10.0
h = 0.001
p = 0.5
truncCutoff = 1.0e-14
dataFilename = joinpath("..", "data", "com-dblp", "com-dblp.ungraph.txt")
commFilename = joinpath("..", "data", "com-dblp", "com-dblp.all.cmty.txt")
runner.runWithCommGroundTruth(dataFilename, commFilename, h, t, p, truncCutoff)
