const DEFAULT_H_THRESH = 0.2
const DEFAULT_FRAC_THRESH = 0.1
const DEFAULT_VAR = 0.01
# const DEFAULT_TOP_NODE_COUNT = 5


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

function stripe_fraction_to_global(x::AbstractVector, y::AbstractVector{<:AbstractVector},
                                  stg_stn::STGSettings, relevant_T)
    stripe_fraction_to_global(x, y, stg_stn.σ, stg_stn.kernel, stg_stn.TP, stg_stn.condition, relevant_T, stg_stn.input_noise)

end

function stripe_fraction_to_global(x::AbstractVector, y::AbstractVector{<:AbstractVector},
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
    gp_phase_fraction, uncertainty = get_phase_global_data(temperature_processes, Tout)
    return conditions, gp_phase_fraction, uncertainty
end

function get_temperature_process(G::Gaussian, x::AbstractVector,
                                y::AbstractVector, σ::Real,
                                P, T_max::Real, log10_τ::Real,
                                input_noise::Val{false})
    C = conditional(G, x, Gaussian(y, σ^2), tol = 1e-6) # condition Gaussian process in position

    inv_profile = inverse_profile(P, T_max, log10_τ)
    t = input_transformation(C, inv_profile) # transform the input of conditional process
end

function get_temperature_process(G::Gaussian, x::AbstractVector,
                                 y::AbstractVector, σ::AbstractVector,
                                 P, T_max::Real, log10_τ::Real,
                                 input_noise::Val{false})
    @. σ = max(σ, DEFAULT_VAR)
    C = conditional(G, x, Gaussian(y, diagm(σ).^2), tol = 1e-6) # condition Gaussian process in position

    inv_profile = inverse_profile(P, T_max, log10_τ)
    t = input_transformation(C, inv_profile) # transform the input of conditional process
end

# same as above but takes into account the uncertainty in the temperature
function get_temperature_process(G::Gaussian, x::AbstractVector,
                                y::AbstractVector, σ_out,
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
        D = GaussianDistributions.derivative(C) # take the derivative w.r.t. T
        DT[:, i] = mean(D).(unit_outer_temperature) # record mean and var of derivative of each process
        UT[:, i] = var(D).(unit_outer_temperature)
    end
    # euclidean norm of temperature gradients of optical coefficients
    d, u = zeros(nout), zeros(nout)
    for i in 1:nout
        di, ui = @views DT[i, :], UT[i, :]
        d[i] = norm(di) # and first-order uncertainty propagation through "norm",
        u[i] = dot(ForwardDiff.gradient(norm, di).^2, ui)
    end
    # @. u = sqrt(u) # convert variance to std?
    return d, u # gradient values and their uncertainties
end

function get_phase_global_data(temperature_processes::AbstractVector{<:Gaussian},
                            outer_temperature::AbstractVector{<:Real},
                            temperature_domain::NTuple{2, <:Real} = (0, 1400))
    ut = unit_transform(temperature_domain...)
    iut = inv_unit_transform(temperature_domain...)
    unit_outer_temperature = ut.(outer_temperature)
    nout = length(outer_temperature)
    DT = zeros(nout, length(temperature_processes))
    UT = similar(DT) # uncertainty
    for i in eachindex(temperature_processes) # for (i, G) in enumerate(temperature_processes)
        C = input_transformation(temperature_processes[i], iut) # inverse of temperature unit-scaling
        # D = GaussianDistributions.derivative(C) # take the derivative w.r.t. T
        DT[:, i] = mean(C).(unit_outer_temperature) # record mean and var of derivative of each process
        UT[:, i] = var(C).(unit_outer_temperature)
    end
    return DT, UT
end

function stripe_entropy_to_global(x::AbstractVector, y::AbstractVector,
                               stg_stn::STGSettings, relevant_T,
                               temperature_domain::NTuple{2, <:Real} = (0, 1400))
    stripe_entropy_to_global(x, y,
                            stg_stn.σ, stg_stn.kernel, stg_stn.TP, stg_stn.condition,
                            relevant_T, stg_stn.input_noise, temperature_domain)
end

function stripe_entropy_to_global(x::AbstractVector, y::AbstractVector,
                                    y_uncer, stg_stn::STGSettings, relevant_T,
                                    temperature_domain::NTuple{2, <:Real} = (0, 1400))
    stripe_entropy_to_global(x, y,
            y_uncer, stg_stn.kernel, stg_stn.TP, stg_stn.condition,
            relevant_T, stg_stn.input_noise, temperature_domain)
end

function stripe_entropy_to_global(x::AbstractVector, y::AbstractVector,
                                σ, k, P, condition::NTuple,
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
                         ts_stn::AbstractTreeSearchSettings,
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
                        ts_stn::AbstractTreeSearchSettings,
                        stg_stn::STGSettings,
                        relevant_T)
    act, act_uncer, Ws, Hs, nodes_for_entropy_calculation, probability_for_entropy_calculation = get_unnormalized_phase_fractions(q, Y, cs,ts_stn=ts_stn, stg_stn=stg_stn)
    entropy = get_expected_entropy(Ws, Hs, act[:,end], nodes_for_entropy_calculation, probability_for_entropy_calculation, length(cs))
    gp_act = Vector{Vector{Float64}}()
    gp_act_uncer = Vector{Vector{Float64}}()
    for i in axes(act, 2)
        _, smoothed_act, smoothed_act_uncer = stripe_entropy_to_global(x, act[:, i], act_uncer[:, i], stg_stn, relevant_T)
        push!(gp_act, smoothed_act)
        push!(gp_act_uncer, smoothed_act_uncer)
    end
    gp_act = transpose(permutedims(hcat(gp_act...)))
    gp_act_uncer = transpose(permutedims(hcat(gp_act_uncer...)))
    phase_fraction = normalize_with_amorphous!(act)
    return stripe_entropy_to_global(x, entropy, stg_stn, relevant_T)..., gp_act, gp_act_uncer, phase_fraction
end


function expected_fraction_to_global(x::AbstractVector, q::AbstractVector, Y::AbstractMatrix,
                                                cs::AbstractVector{<:CrystalPhase},
                                                ts_stn::AbstractTreeSearchSettings,
                                                stg_stn::STGSettings,
                                                relevant_T)
    act, act_uncer, Ws, Hs, nodes_for_entropy_calculation, probability_for_entropy_calculation = get_unnormalized_phase_fractions(q, Y, cs,ts_stn=ts_stn, stg_stn=stg_stn)
    expected_fracs = get_expected_fraction(Ws, Hs, act[:,end], nodes_for_entropy_calculation, probability_for_entropy_calculation, length(cs))
    phase_fraction = normalize_with_amorphous!(expected_fracs)
    return stripe_fraction_to_global(x, [ phase_fraction[:,i] for i in axes(phase_fraction, 2)], stg_stn, relevant_T)..., phase_fraction, nodes_for_entropy_calculation, probability_for_entropy_calculation, Ws
end


# Simple version, not doing refinement
function get_unnormalized_phase_fractions(x, Y, Y_uncer, cs; ts_stn::AbstractTreeSearchSettings, stg_stn::STGSettings) # Add background as input
    pr = ts_stn.opt_stn.priors
    W, H, K = xray(Array(transpose(Y)), stg_stn.nmf_rank)
    best_result_nodes = Vector{Node}(undef, stg_stn.nmf_rank-1)
    best_result_node_prob = Vector{Float64}(undef, stg_stn.nmf_rank-1)
    nodes_for_entropy_calculation = Array{Node, 2}(undef, (stg_stn.nmf_rank-1, stg_stn.n_top_node))
    probability_for_entropy_calculation = zeros(Float64, (stg_stn.nmf_rank-1, stg_stn.n_top_node))

    amorphous_idx, amorphous = classify_amorphous(W, H)
    # amorphous -= background
    Ws = @view W[: ,filter(!=(amorphous_idx), 1:size(W, 2))]
    Hs = @view H[filter(!=(amorphous_idx), 1:size(H, 1)), :]
    y_uncers = @view Y_uncer[K[filter(!=(amorphous_idx), 1:length(K))], :]

    phase_frac_of_bases = zeros(Measurement{Float64}, (size(Hs, 1), length(cs)))
    amorphous_frac = zeros(Float64, size(Ws, 2))
    fractions = zeros(Measurement{Float64}, (size(Y, 1), length(cs)+1))
    fractions[:,end] += H[amorphous_idx, :]

    # TODO: Real background estimation to separate amorphous from MCBL results
    for i in axes(Ws, 2)
        if !(stg_stn.check_amorphous && is_amorphous(x, Ws[:, i], stg_stn.background_length, 10.))
            # temperaly using background; Should use root node for amorphous determination
            # Background subtraction
            b = mcbl(Ws[:, i], x, stg_stn.background_length)
            # amorphous_bg = b - background

            amorphous_scale = scaling_fit(amorphous, b, [1.0])
            amorphous_frac[i] = amorphous_scale[1]

            # plt = plot(x, Ws[:,i]/maximum(Ws[:,i]))
            Ws[:, i] -= b
            y_uncer = y_uncers[i, :] / maximum(Ws[:, i])
            y = Ws[:,i] / maximum(Ws[:, i])
            # plot!(x, Ws[:,i]/maximum(Ws[:,i]), title="$(i)")

            # Tree search
            if ts_stn isa TreeSearchSettings
                lt = Lazytree(cs, x) # 5 is just random number that is not used
                result = search!(lt, x, y, y_uncer, ts_stn)
            elseif ts_stn isa MPTreeSearchSettings
                mpt = MPTree(cs, x, 100, 0.1)
                result = search!(mpt, x, y, y_uncer, ts_stn)
            else
                error("TreeSearchSetting provided is not supported yet")
            end

            results = reduce(vcat, result)
            # probs = get_probabilities(results, x, y, y_uncer, pr.std_noise, pr.mean_θ, pr.std_θ,normalization_constant= ts_stn.normalization_constant)
            probs = get_probabilities(results, x, y, y_uncer, pr.mean_θ, pr.std_θ,normalization_constant= ts_stn.normalization_constant)

            # TODO: Need to do probability check here. How do we flag potentially unidentifiable phases or amorphous
            # Take top-x probabilities and do a thresholding
            best_result_node = results[argmax(probs)]

            prob_permute = sortperm(probs, rev=true)
            best_result_nodes_ = results[prob_permute[1:stg_stn.n_top_node]]

            if !isempty(stg_stn.save_plot)
                for node_idx in eachindex(best_result_nodes_)
                    plt = plot(x, y, label="XRD Pattern", xlabel="q (nm⁻¹)", ylabel="Normalized Intensity", title="basis_$(i)_top_$(node_idx) Prob $(probs[prob_permute[node_idx]])")
                    for phase_idx in eachindex(best_result_nodes_[node_idx].phase_model.CPs)
                        plot!(x, evaluate!(zero(x), best_result_nodes_[node_idx].phase_model.CPs[phase_idx], x), label=best_result_nodes_[node_idx].phase_model.CPs[phase_idx].name)
                    end
                    savefig("$(stg_stn.save_plot)_basis_$(i)_top_$(node_idx)_node.png")
                end
            end

            # plt = plot(x, evaluate!(zero(x), best_result_nodes_[1].phase_model, x ), title= "$(std(y.-evaluate!(zero(x), best_result_node.phase_model, x )))")
            # display(plt)
            best_probs_ = probs[prob_permute[1:stg_stn.n_top_node]]
            for i in 1:stg_stn.n_top_node
                println("###")
                println("Top $(i) node, Probability: $(best_probs_[i])")
                println(best_result_nodes_[i].phase_model)
            end
            println("###")

            top_prob = sort(probs, rev=true)[1:stg_stn.n_top_node]

            if reject_probs(top_prob)
                # println(top_prob)
                println("rejected")
                continue
            end

            nodes_for_entropy_calculation[i, :] = results[sortperm(probs, rev=true)[1:stg_stn.n_top_node]]
            probability_for_entropy_calculation[i, :] = sort(probs, rev=true)[1:stg_stn.n_top_node]
            best_result_node_prob[i] = maximum(probs)
            best_result_nodes[i] = best_result_node
            phase_frac_of_bases[i, :] = get_phase_ratio_with_uncertainty(phase_frac_of_bases[i, :],
                                                                        best_result_node.phase_model.CPs,
                                                                        x, y, ts_stn.opt_stn)
        else
            # Count as amorphous
            fractions[1:end, end] += Hs[i,:]
        end
    end

    frac, frac_uncer = calculate_unnormalized_fractions!(fractions, Ws, Hs, phase_frac_of_bases, amorphous_frac,
                                                        best_result_nodes, stg_stn.h_threshold, stg_stn.frac_threshold)
    # return all nodes (vector of vector)
    frac, frac_uncer, Ws, Hs, nodes_for_entropy_calculation, probability_for_entropy_calculation
end

function get_unnormalized_phase_fractions(x, Y, cs; ts_stn::AbstractTreeSearchSettings, stg_stn::STGSettings) 
    Y_uncer = zero(Y)
    get_unnormalized_phase_fractions(x, Y, Y_uncer, cs, ts_stn=ts_stn, stg_stn=stg_stn)
end

function get_phase_ratio_with_uncertainty(fraction::AbstractVector, CPs::AbstractVector, x::AbstractVector, y::AbstractVector,
                                        opt_stn::OptimizationSettings, scaled::Bool=false)

    act_ind = get_activation_indicies(length(CPs))
    σ_ind = get_σ_indicies(length(CPs))
    uncer = uncertainty(CPs, x, y, opt_stn, scaled)
    act_var = uncer[act_ind]
    σ_var = uncer[σ_ind]

    act = Vector{Measurement}(undef, length(CPs))
    width = similar(act)# Vector{Measurement}(undef, length(ind))
    for i in eachindex(act_var)
        if act_var[i] > 0.
            log_act = log(CPs[i].act)
            act[i] = exp(log_act ± sqrt(act_var[i]))
            log_σ = log(CPs[i].σ)
            width[i] = exp(log_σ ± sqrt(σ_var[i]))

        else
            # IDEA: Should we flag this???
            act[i] = CPs[i].act ± CPs[i].act # This means its not at an local minimum
            width[i] = CPs[i].σ ± CPs[i].σ
        end
    end

    for i in eachindex(CPs)
        fraction[CPs[i].id+1] += act[i] * get_n(CPs[i].profile, width[i]) / CPs[i].norm_constant
    end

    fraction
end

get_activation_indicies(num_phase::Integer) = [7 + 8*(i-1) for i in 1:num_phase]
get_σ_indicies(num_phase::Integer) = [8 + 8*(i-1) for i in 1:num_phase]