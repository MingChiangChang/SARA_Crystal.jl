using NPZ
using CrystalShift
using CrystalShift: FixedPseudoVoigt, OptimizationSettings

using CrystalTree: TreeSearchSettings

using SARA_Crystal: get_phase_fractions, phase_to_global, TemperatureProfile, get_relevant_T
using SARA_Crystal: entropy_to_global, STGSettings
using CovarianceFunctions


path = "/Users/ming/Desktop/Code/XRD_paper/data/"
q_path = path * "TaSnO_q.npy"
data_path = path * "TaSnO_data.npy"
stick_path = path * "TaSnO_sticks.csv"

q = npzread(q_path)[336,100:800]
data = npzread(data_path)[336,:,100:800]

f = open(stick_path, "r")
if Sys.iswindows()
    s = split(read(f, String), "#\r\n") # Windows: #\r\n ...
else
    s = split(read(f, String), "#\n")
end

if s[end] == ""
    pop!(s)
end

cs = Vector{CrystalPhase}(undef, size(s))
cs = @. CrystalPhase(String(s), (0.1, ), (FixedPseudoVoigt(0.4), ))
# cs = cs[[1, 5, 10, 11, 12]]
tp = TemperatureProfile()
# @time t = get_phase_fractions(q, data, cs;
#                         rank = 4,
#                         length_scale=8.,
#                         depth=2,
#                         s,
#                         k=5,
#                         std_noise=1.,
#                         mean_θ=[1., .5, 1.],
#                         std_θ=[0.05, 10., 1.],
#                         maxiter=128,
#                         h_threshold=0.01,
#                         frac_threshold=0.1)
kernel = CovarianceFunctions.MaternP(2)
kernel = CovarianceFunctions.Lengthscale(kernel, 0.05)
nout = 32 # we can still change this after the creation of the data
# constant_offset = Val(false)
# T_proportions = (.75, .99) # as a proportion of T_max, what data to convert to gradients
constant_offset_bool = true
constant_offset = Val(constant_offset_bool)
T_offset = [200, 10] # number of degrees from T_max we are generating data for
relevant_T = get_relevant_T(constant_offset, T_offset..., nout)
x = collect(-1.:.01:1.)
rank = 4
h_threshold = .1
frac_threshold = .1
condition = (1300, 3)

opt_stn = OptimizationSettings{Float64}(0.1, [1., .5, .5], [0.05, 10., 1.], 128, true, LM, "LS", Simple)
stg_stn = STGSettings(rank, h_threshold, frac_threshold, 8., kernel, 0.05, tp, condition, Val(false))
ts_stn = TreeSearchSettings{Float64}(2, 3, opt_stn)
# @time t = phase_to_global(x, q, data, cs;
#                         rank =4,
#                         length_scale=8.,
#                         depth=2,
#                         s,
#                         search_k=3,
#                         std_noise=0.1,
#                         mean_θ=[1., .5, .5],
#                         std_θ=[0.05, 10., 1.],
#                         maxiter=128,
#                         h_threshold=0.1,
#                         frac_threshold=0.1,
#                         σ=0.05, kernel=kernel,
#                         P=tp, condition=(1300, 3), relevant_T=relevant_T)
@time t = phase_to_global(x, q, data, cs, ts_stn, stg_stn, relevant_T)