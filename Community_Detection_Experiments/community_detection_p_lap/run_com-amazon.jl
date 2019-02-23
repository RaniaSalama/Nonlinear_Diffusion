include("run_against_ground_truth.jl")
using DiffusionRunner
runner = DiffusionRunner

t = 10.0
h = 0.001
p = 1.9
truncCutoff = 1.0e-16
dataFilename = joinpath("..", "data", "amazon", "com-amazon.ungraph.txt")
commFilename = joinpath("..", "data", "amazon", "com-amazon.all.dedup.cmty.txt")
runner.runWithCommGroundTruth(dataFilename, commFilename, h, t, p, truncCutoff)
