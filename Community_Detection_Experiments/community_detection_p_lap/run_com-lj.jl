include("run_against_ground_truth.jl")
using DiffusionRunner
runner = DiffusionRunner

t = 10.0
h = 0.001
p = 1.9
truncCutoff = 1.0e-10
dataFilename = joinpath("..", "data", "com-lj", "com-lj.ungraph.txt")
commFilename = joinpath("..", "data", "com-lj", "com-lj.all.cmty.txt")
runner.runWithCommGroundTruth(dataFilename, commFilename, h, t, p, truncCutoff)

