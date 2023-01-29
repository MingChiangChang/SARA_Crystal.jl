using SARA_Crystal: scaling_fit
# stn = LevenbergMarquartSettings(min_resnorm = 1e-2, min_res = 1e-10,
# 						min_decrease = 1e-6, min_iter=10, max_iter = 16,
# 						decrease_factor = 7, increase_factor = 10, max_step = 0.1)

# stn = LevenbergMarquartSettings()
x = collect(-pi:.1:pi)
y = 5.2*sin.(x) + rand(length(x)) .+ 5
base = sin.(x)

#function t(r, p)
#    @. r = y - base*p[1] + p[2]
#    println(sum(abs2, r))
#    r
#end
#
#
#θ = [0.5, 1.0]
#lm = LevenbergMarquart(t, θ, zero(y))
#params, iter = OptimizationAlgorithms.optimize!(lm, θ, zero(y), stn, 1., Val(true))

params = scaling_fit(base, y, [1.0])