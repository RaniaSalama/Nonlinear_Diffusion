include("run_against_ground_truth.jl")
using DiffusionRunner
runner = DiffusionRunner

t = 10.0
h = 0.001
p = 1.9
truncCutoff = 1.0e-9
dataFilename = joinpath("..", "..", "data", "com-youtube", "com-youtube.ungraph.txt")
commFilename = joinpath("..", "..", "data", "com-youtube", "com-youtube.all.cmty.txt")
runner.runWithCommGroundTruth(dataFilename, commFilename, h, t, p, truncCutoff)


# 1e-7:  mean fmeas=0.4669        mean setsize=34.8600    mean cond=0.6825
# 1e-8:  mean fmeas=0.4669        mean setsize=34.8600    mean cond=0.6825
# 1e-9: mean fmeas=0.4216        mean setsize=215.8200   mean cond=0.6552
# 1e-10: mean fmeas=0.3833        mean setsize=1531.2400          mean cond=0.6285