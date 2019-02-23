include("run_against_ground_truth.jl")
using DiffusionRunner
runner = DiffusionRunner

t = 10.0
h = 0.001
p = 0.5
truncCutoff = 2.0e-8
dataFilename = joinpath("..", "data", "amazon", "com-amazon.ungraph.txt")
commFilename = joinpath("..", "data", "amazon", "com-amazon.all.dedup.cmty.txt")
runner.runWithCommGroundTruth(dataFilename, commFilename, h, t, p, truncCutoff)