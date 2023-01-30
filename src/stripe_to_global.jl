const DEFAULT_H_THRESH = 0.2
const DEFAULT_FRAC_THRESH = 0.1


# Process single stripe to global infomation
# For 2D case
function stripe_to_global(x::AbstractVector, y::AbstractVector{<:AbstractVector},
                        σ::Real, k, P, T_max::Real, log10_τ::Real,
                        relevant_T, input_noise::Union{Val{true}, Val{false}} = Val(true))
    stripe_to_global(x, y, σ, k, P, (T_max, log10_τ), relevant_T, input_noise)[2:3]
end

# For potentially 3D or higher dimensional case
# condition is tuple of (T_max, τ, composition, ...) conditions
# Note: x is in mm
# Return:
#   conditions: vector of T_peak, dwell and compositions
#   phases: vector of vector. Each vector contains phase fraction of candidate phases
#   uncertainty:

function stripe_to_global(x::AbstractVector, y::AbstractVector{<:AbstractVector}, stg_stn::STGSettings, relevant_T)
    stripe_to_global(x, y, stg_stn.σ, stg_stn.kernel, stg_stn.TP, stg_stn.condition, relevant_T, stg_stn.input_noise)
end

function stripe_to_global(x::AbstractVector, y::AbstractVector{<:AbstractVector},
                            σ::Real, k, P, condition::NTuple,
                            relevant_T, input_noise::Union{Val{true}, Val{false}} = Val(true))
    all(==(length(x)), length.(y)) || throw(DimensionMismatch("x and elements of y do not have same lengths: length(x) = $(length(x)) and length.(y) = $(length.(y))"))
    T_max, log10_τ = condition[1:2]
    other_conditions = condition[3:end]
    G = Gaussian(k) # define prior GP
    temperature_processes = Vector{Gaussian}([G for _ in 1:length(y)])
    for j in eachindex(y)
        C = get_temperature_process(G, x, y[j], σ, P, T_max, log10_τ, input_noise)
        temperature_processes[j] = C
    end
    # C = temperature_processes[1]
    Tout = relevant_T(T_max, log10_τ) # IDEA: could add composition as dimension
    conditions = tuple.(Tout, log10_τ, other_conditions...)
    gradients, uncertainty = get_global_data(temperature_processes, Tout)
    return conditions, gradients, uncertainty
end

function get_temperature_process(G::Gaussian, x::AbstractVector,
                                y::AbstractVector, σ::Real,
                                P, T_max::Real, log10_τ::Real,
                                input_noise::Val{false})
    C = conditional(G, x, Gaussian(y, σ^2), tol = 1e-6) # condition Gaussian process in position

    inv_profile = inverse_profile(P, T_max, log10_τ)
    t = input_transformation(C, inv_profile) # transform the input of conditional process
end

# same as above but takes into account the uncertainty in the temperature
function get_temperature_process(G::Gaussian, x::AbstractVector,
                                y::AbstractVector, σ_out::Real,
                                P, T_max::Real, log10_τ::Real,
                                input_noise::Val{true})
    # first, get regular GP w.r.t. T
    C = get_temperature_process(G, x, y, σ_out, P, T_max, log10_τ, Val(false))
    # with input noise, adjust GP with generalized NIGP model
    T = profile(P, T_max, log10_τ).(x) # maps position of measured optical data to temperature
    d = (t->ForwardDiff.derivative(mean(C), t)).(T)
    var_in = temperature_uncertainty(P, T_max, log10_τ).(x) # input uncertainty corresponding to positions
    Σ = @. var_in * d^2 + σ_out^2
    Σ = Diagonal(Σ)
    C = conditional(G, x, Gaussian(y, Σ), tol = 1e-6) # calculate new conditional process with adjusted noise variance
    inv_profile = inverse_profile(P, T_max, log10_τ)
    input_transformation(C, inv_profile) # transform the input to temperature domain
end

function get_global_data(temperature_processes::AbstractVector{<:Gaussian},
                        outer_temperature::AbstractVector{<:Real},
                        temperature_domain::NTuple{2, <:Real} = (0, 1400))
    # normalize temperature input to have a normalized scale for derivatives
    ut = unit_transform(temperature_domain...)
    iut = inv_unit_transform(temperature_domain...)
    unit_outer_temperature = ut.(outer_temperature)
    nout = length(outer_temperature)
    DT = zeros(nout, length(temperature_processes))
    UT = similar(DT) # uncertainty
    for i in eachindex(temperature_processes) # for (i, G) in enumerate(temperature_processes)
        C = input_transformation(temperature_processes[i], iut) # inverse of temperature unit-scaling
        # plt = plot(mean(C).(unit_outer_temperature))
        # display(plt)
        D = GaussianDistributions.derivative(C) # take the derivative w.r.t. T
        DT[:, i] = mean(D).(unit_outer_temperature) # record mean and var of derivative of each process
        UT[:, i] = var(D).(unit_outer_temperature)
    end
    # euclidean norm of temperature gradients of optical coefficients
    # println(unit_outer_temperature)
    # plt = heatmap(DT)
    # display(plt)
    d, u = zeros(nout), zeros(nout)
    for i in 1:nout
        di, ui = @views DT[i, :], UT[i, :]
        d[i] = norm(di) # and first-order uncertainty propagation through "norm",
        u[i] = dot(ForwardDiff.gradient(norm, di).^2, ui)
    end
    # @. u = sqrt(u) # convert variance to std?
    return d, u # gradient values and their uncertainties
end

function stripe_entropy_to_global(x::AbstractVector, y::AbstractVector,
                               stg_stn::STGSettings, relevant_T,
                               temperature_domain::NTuple{2, <:Real} = (0, 1400))
    stripe_entropy_to_global(x, y,
                            stg_stn.σ, stg_stn.kernel, stg_stn.TP, stg_stn.condition,
                            relevant_T, stg_stn.input_noise, temperature_domain)
end

function stripe_entropy_to_global(x::AbstractVector, y::AbstractVector,
                                σ::Real, k, P, condition::NTuple,
                                relevant_T, input_noise::Union{Val{true}, Val{false}} = Val(true),
                                temperature_domain::NTuple{2, <:Real} = (0, 1400))
    all(==(length(x)), length(y)) || throw(DimensionMismatch("x and elements of y do not have same lengths: length(x) = $(length(x)) and length.(y) = $(length.(y))"))
    T_max, log10_τ = condition[1:2]
    other_conditions = condition[3:end]
    G = Gaussian(k) # define prior GP
    C = get_temperature_process(G, x, y, σ, P, T_max, log10_τ, input_noise)
    Tout = relevant_T(T_max, log10_τ) # IDEA: could add composition as dimension
    conditions = tuple.(Tout, log10_τ, other_conditions...)
    entropy, uncertainty = get_global_entropy(C, Tout, temperature_domain)
    conditions, entropy, uncertainty
end

function get_global_entropy(temperature_process::Gaussian,
                            outer_temperature::AbstractVector{<:Real},
                            temperature_domain::NTuple{2, <:Real} = (0, 1400))
    DT = zeros(length(outer_temperature))
    UT = similar(DT)
    ut = unit_transform(temperature_domain...)
    iut = inv_unit_transform(temperature_domain...)
    unit_outer_temperature = ut.(outer_temperature)
    C = input_transformation(temperature_process, iut)
    entropy = mean(C).(unit_outer_temperature)
    uncertainty = var(C).(unit_outer_temperature)
    entropy, uncertainty
end

function phase_to_global(x::AbstractVector, q::AbstractVector, Y::AbstractMatrix,
                         cs::AbstractVector{<:CrystalPhase},
                         ts_stn::CrystalTree.TreeSearchSettings,
                         stg_stn::STGSettings,
                         relevant_T)
    y = get_phase_fractions(q, Y, cs, ts_stn=ts_stn, stg_stn=stg_stn)
    renormalize!(y)
    stripe_to_global(x, [y[:,i] for i in 1:axes(y, 2)], stg_stn, relevant_T)
end



function phase_to_global(x::AbstractVector, q::AbstractVector, Y::AbstractMatrix,
                         cs::AbstractVector{<:CrystalPhase};
                         rank::Int,
                         length_scale::Real,
                         depth::Int,
                         search_k::Int,
                         std_noise::Real,
                         mean_θ::AbstractVector,
                         std_θ::AbstractVector,
                         maxiter::Int,
                         h_threshold::Real,
                         frac_threshold::Real,
                         σ, kernel,
                         P,
                         condition::NTuple,
                         relevant_T,
                         input_noise::Union{Val{true}, Val{false}} = Val(false))

    opt_stn = OptimizationSettings{Float64}(std_noise, mean_θ, std_θ, maxiter, true, LM, "LS", Simple)
    ts_stn = TreeSearchSettings(depth, search_k, opt_stn)
    stg_stn = STGSettings(rank, h_threshold, frac_threshold, length_scale, kernel, σ, P, condition, input_noise)
    y = get_phase_fractions(q, Y, cs;
                            ts_stn = ts_stn, stg_stn=stg_stn)
    # plt = heatmap(y)
    # display(plt)
    stripe_to_global(x, [y[:,i] for i in axes(y, 2)], σ, kernel, P, condition, relevant_T, input_noise)
end

function entropy_to_global(x::AbstractVector, q::AbstractVector, Y::AbstractMatrix,
                        cs::AbstractVector{<:CrystalPhase},
                        ts_stn::CrystalTree.TreeSearchSettings,
                        stg_stn::STGSettings,
                        relevant_T)
    fractions, fraction_uncer = get_phase_fractions(q, Y, cs,ts_stn=ts_stn, stg_stn=stg_stn)
    # entropy_renormalize!(y)
    entropy = get_entropy(fractions)
    return stripe_entropy_to_global(x, entropy, stg_stn, relevant_T)..., fractions, fraction_uncer
end


# Simple version, not doing refinement
function get_phase_fractions(x, Y, cs; ts_stn::TreeSearchSettings, stg_stn::STGSettings)
    pr = ts_stn.opt_stn.priors
    W, H, _ = xray(Array(transpose(Y)), stg_stn.nmf_rank)
    result_nodes = Vector{Node}(undef, stg_stn.nmf_rank-1)
    result_node_prob = Vector{Float64}(undef, stg_stn.nmf_rank-1)

    amorphous_idx, amorphous = classify_amorphous(W, H)
    Ws = @view W[: ,filter(!=(amorphous_idx), 1:size(W, 2))]
    Hs = @view H[filter(!=(amorphous_idx), 1:size(H, 1)), :]

    phase_frac_of_bases = zeros(Measurement{Float64}, (size(Hs, 1), length(cs)))
    amorphous_frac = zeros(Float64, size(Ws, 2))
    fractions = zeros(Measurement{Float64}, (size(Y, 1), length(cs)+1))
    fractions[:,end] += H[amorphous_idx, :]

    # TODO: Real background estimation to separate amorphous from MCBL results
    for i in axes(Ws, 2)
        if !is_amorphous(x, Ws[:, i], stg_stn.background_length, 10.) # temperal; Should use root node for amorphous determination
            # Background subtraction
            b = mcbl(Ws[:, i], x, stg_stn.background_length)

            amorphous_scale = scaling_fit(amorphous, b, [1.0])
            amorphous_frac[i] = amorphous_scale[1]

            Ws[:, i] -= b
            y = Ws[:,i] / maximum(Ws[:, i])

            # Tree search
            lt = Lazytree(cs, x, 5) # 5 is just random number that is not used
            result = search!(lt, x, y, ts_stn)
            results = reduce(vcat, result)
            probs = get_probabilities(results, x, y, pr.std_noise, pr.mean_θ, pr.std_θ)
            # TODO: Need to do probability check here. How do we flag potentially unidentifiable phases or amorphous
            result_node = results[argmax(probs)]
            result_node_prob[i] = maximum(probs)
            result_nodes[i] = result_node
            phase_frac_of_bases[i, :] = get_phase_ratio_with_uncertainty(phase_frac_of_bases[i, :],
                                                                        result_node.phase_model.CPs,
                                                                        x, y, ts_stn.opt_stn)
        else
            # Count as amorphous
            fractions[1:end] += Hs[i,:]
        end
    end

    frac, frac_uncer = calculate_unnormalized_fractions!(fractions, Ws, Hs, phase_frac_of_bases, amorphous_frac,
                                                        result_nodes, stg_stn.h_threshold, stg_stn.frac_threshold)
    frac, frac_uncer
end

function get_phase_ratio_with_uncertainty(fraction::AbstractVector, CPs::AbstractVector, x::AbstractVector, y::AbstractVector,
                                        opt_stn::OptimizationSettings, scaled::Bool=false)

    ind = get_activation_indicies(length(CPs))
    act_uncer = uncertainty(CPs, x, y, opt_stn, scaled)[ind]
    log_act = log.([CP.act for CP in CPs])
    act = exp.(log_act .± act_uncer)

    for i in eachindex(CPs)
        fraction[CPs[i].id+1] += act[i] / CPs[i].norm_constant
    end

    fraction
end

get_activation_indicies(num_phase::Integer) = [7 + 8*(i-1) for i in 1:num_phase]