const DEFAULT_RES_THRESH = 1.
const DEFAULT_H_THRESH = 0.2
const DEFAULT_FRAC_THRESH = 0.1


# Process single stripe to global infomation
# For 2D case
function stripe_to_global(x::AbstractVector, y::AbstractVector{<:AbstractVector},
                        σ::Real, k, P::TemperatureProfile, T_max::Real, log10_τ::Real,
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
using Statistics: mean
function stripe_to_global(x::AbstractVector, y::AbstractVector{<:AbstractVector}, stg_stn::STGSettings, relevant_T)
    stripe_to_global(x, y, stg_stn.σ, stg_stn.kernel, stg_stn.TP, stg_stn.condition, relevant_T, stg_stn.input_noise)
end

function stripe_to_global(x::AbstractVector, y::AbstractVector{<:AbstractVector},
                            σ::Real, k, P::TemperatureProfile, condition::NTuple,
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
                                P::TemperatureProfile, T_max::Real, log10_τ::Real,
                                input_noise::Val{false})
    C = conditional(G, x, Gaussian(y, σ^2), tol = 1e-6) # condition Gaussian process in position
    plt = plot(x, mean(C).(x), ribbon=var(C).(x))
    display(plt)

    inv_profile = inverse_profile(P, T_max, log10_τ)
    t = input_transformation(C, inv_profile) # transform the input of conditional process
end

# same as above but takes into account the uncertainty in the temperature
function get_temperature_process(G::Gaussian, x::AbstractVector,
                                y::AbstractVector, σ_out::Real,
                                P::TemperatureProfile, T_max::Real, log10_τ::Real,
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
    plt = heatmap(DT)
    display(plt)
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
                                σ::Real, k, P::TemperatureProfile, condition::NTuple,
                                relevant_T, input_noise::Union{Val{true}, Val{false}} = Val(true),
                                temperature_domain::NTuple{2, <:Real} = (0, 1400))
    all(==(length(x)), length(y)) || throw(DimensionMismatch("x and elements of y do not have same lengths: length(x) = $(length(x)) and length.(y) = $(length.(y))"))
    T_max, log10_τ = condition[1:2]
    other_conditions = condition[3:end]
    G = Gaussian(k) # define prior GP
    C = get_temperature_process(G, x, y, σ, P, T_max, log10_τ, input_noise)
    Tout = relevant_T(T_max, log10_τ) # IDEA: could add composition as dimension
    conditions = tuple.(Tout, log10_τ, other_conditions...)
    entropy, uncertainty = get_global_entropy(C, Tout)
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
                         cs::AbstractVector{<:CrystalPhase}, s;
                         ts_stn::CrystalTree.TreeSearchSettings,
                         stg_stn::STGSettings,
                         relevant_T)
    println("called")
    y = get_phase_fractions(q, Y, cs, s, ts_stn=ts_stn, stg_stn=stg_stn)
    stripe_to_global(x, [y[:,i] for i in 1:size(y,2)], stg_stn, relevant_T)
end



function phase_to_global(x::AbstractVector, q::AbstractVector, Y::AbstractMatrix,
                         cs::AbstractVector{<:CrystalPhase};
                         rank::Int,
                         length_scale::Real,
                         depth::Int,
                         s,
                         search_k::Int,
                         std_noise::Real,
                         mean_θ::AbstractVector,
                         std_θ::AbstractVector,
                         maxiter::Int,
                         h_threshold::Real,
                         frac_threshold::Real,
                         σ, kernel,
                         P::TemperatureProfile,
                         condition::NTuple,
                         relevant_T,
                         input_noise::Union{Val{true}, Val{false}} = Val(false))

    y = get_phase_fractions(q, Y, cs;
                            rank = rank,
                            length_scale=length_scale,
                            depth = depth,
                            s = s,
                            k = search_k,
                            std_noise = std_noise,
                            mean_θ = mean_θ,
                            std_θ = std_θ,
                            maxiter = maxiter,
                            h_threshold = h_threshold,
                            frac_threshold  = frac_threshold)
    plt = heatmap(y)
    display(plt)
    stripe_to_global(x, [y[:,i] for i in 1:size(y,2)], σ, kernel, P, condition, relevant_T, input_noise)
end

function entropy_to_global(x::AbstractVector, Y::AbstractMatrix,
                            cs::AbstractVector{<:CrystalPhase};
                            rank::Int,
                            length_scale::Real,
                            depth::Int,
                            s,
                            search_k::Int,
                            std_noise::Real,
                            mean_θ::AbstractVector,
                            std_θ::AbstractVector,
                            maxiter::Int,
                            h_threshold::Real,
                            frac_threshold::Real,
                            σ, kernel,
                            P::TemperatureProfile,
                            condition::NTuple,
                            relevant_T,
                            input_noise::Union{Val{true}, Val{false}} = Val(false))
    y = get_phase_fractions(x, Y, cs;
                            rank = rank,
                            length_scale=length_scale,
                            depth = depth,
                            s = s,
                            k = search_k,
                            std_noise = std_noise,
                            mean_θ = mean_θ,
                            std_θ = std_θ,
                            maxiter = maxiter,
                            h_threshold = h_threshold,
                            frac_threshold  = frac_threshold)
    plt = heatmap(y)
    display(plt)
    for i in 1:size(y, 1)
        if(any(x -> x>0, y[i,:]))
            y[i,:] ./= sum(y[i, :])
        else
            y[i, :] .= 1/size(y, 2)
        end
    end
    plt = heatmap(y)
    display(plt)
    entropy = get_entropy(y)
    plt = plot(entropy)
    display(plt)
    stripe_entropy_to_global(collect(-1.:.01:1.), entropy, σ, kernel, P, condition, relevant_T, input_noise)
end

get_entropy(p::Real) = p == 0 ? 0 : - p * log(p)
get_entropy(P::AbstractArray) = sum(get_entropy.(P))
get_entropy(P::AbstractMatrix) = reduce(vcat, sum(get_entropy.(P), dims=2))


# Simple version, not doing refinement
function get_phase_fractions(x, Y, cs, s; ts_stn::TreeSearchSettings, stg_stn::STGSettings)
    pr = ts_stn.opt_stn.priors
    W, H, _ = xray(Array(transpose(Y)), stg_stn.nmf_rank)
    result_nodes = Vector{Node}(undef, stg_stn.nmf_rank)

    for i in 1:size(W, 2)
        if !is_amorphous(x, W[:, i], stg_stn.background_length, 10.) # temperal; Should use root node for amorphous determination
            # Background subtraction
            b = mcbl(W[:, i], x, stg_stn.background_length)
            W[:, i] -= b
            W[:, i] /= maximum(W[:, i])

            # Tree search
            lt = Lazytree(cs, x, 5, s)
            result = search!(lt, x, W[:, i], ts_stn)
            results = reduce(vcat, result)
            probs = get_probabilities(results, x, W[:, i], pr.std_noise, pr.mean_θ, pr.std_θ)
            result_node = results[argmax(probs)]
            result_nodes[i] = result_node
        end
    end

    fractions = zeros(Float64, (size(Y, 1), length(cs)))
    get_phase_fractions!(fractions, H, result_nodes, stg_stn.h_threshold, stg_stn.frac_threshold)
end

function get_phase_fractions(x::AbstractVector, Y::AbstractMatrix,
                            cs::AbstractVector{<:CrystalPhase};
                            rank::Int,
                            length_scale::Real,
                            depth::Int,
                            s,
                            k::Int,
                            std_noise::Real,
                            mean_θ::AbstractVector,
                            std_θ::AbstractVector,
                            maxiter::Int,
                            h_threshold::Real=DEFAULT_H_THRESH,
                            frac_threshold::Real=DEFAULT_FRAC_THRESH
                            )
    # Use NMF for dimensionality reduction
    # Only perform search on the NMF basis
    # Merge identified phases and apply threshold

    # NMF
    W, H, _ = xray(Array(transpose(Y)), rank)

    # Tree search
    result_nodes = Vector{Node}(undef, rank)
    for i in 1:rank
        # W[:, i] /= maximum(W[:, i])
        if !is_amorphous(x, W[:, i], length_scale, 10.) # temperal; Should use root node for amorphous determination
            # Background subtraction
            plt = plot(x, W[:, i])
            b = mcbl(W[:, i], x, 7.)
            plot!(x, b)
            display(plt)
            W[:, i] -= b
            #W[W[:,i] .< 1e-5, i] .= 1e-5
            W[:, i] /= maximum(W[:, i])

            # Tree search
            lt = Lazytree(cs, depth, x, 5, s)
            result = search!(lt, x, W[:, i], k, std_noise, mean_θ, std_θ,
                             method=LM, objective="LS", maxiter=maxiter,
                             regularization=true)
            results = reduce(vcat, result)
            probs = get_probabilities(results, x, W[:, i], std_noise, mean_θ, std_θ)
            result_node = results[argmax(probs)]
            plt = plot(x, W[:, i])
            plot!(x, evaluate!(zero(x), result_node.phase_model, x))
            display(plt)
            result_nodes[i] = result_node
        end
    end

    fractions = zeros(Float64, (size(Y, 1), length(cs)))
    get_phase_fractions!(fractions, H, result_nodes, h_threshold, frac_threshold)
end

function get_phase_fractions!(fractions::AbstractMatrix,
                              H::AbstractMatrix,
                              result_nodes::AbstractVector,
                              h_thresh::Real,
                              frac_threshold::Real)
    for colindex in 1:size(H, 2)
        for i in eachindex(result_nodes)
            if  isassigned(result_nodes, i) && H[i, colindex] > h_thresh
                fracs = get_fraction(result_nodes[i].phase_model.CPs)
                for phase_idx in eachindex(fracs)
                    if fracs[phase_idx] >= frac_threshold # Threshold can be 0
                        fractions[colindex, result_nodes[i].phase_model.CPs[phase_idx].id] +=  H[i, colindex] * fracs[phase_idx]
                    end
                end
            end
        end
    end
    fractions
end


####### Helper functions
function is_amorphous(x::AbstractVector, y::AbstractVector, l::Real, p::Real,
                      std_noise::Real=0.05,
                      mean_θ::AbstractVector=[1., 1., 1.],
                      std_θ::AbstractVector=[0.05, 0.5, 0.2],
                      maxiter::Int=512,
                      threshold::Real = DEFAULT_RES_THRESH)
    normalized_y =  y./maximum(y)
    bg = BackgroundModel(x, EQ(), l, p)
    amorphous = PhaseModel(nothing, nothing, bg)
    result = optimize!(amorphous, x, normalized_y, std_noise, mean_θ, std_θ, method=LM, objective="LS",
            maxiter=maxiter, optimize_mode=Simple, regularization=true, verbose=false)
    println(norm(normalized_y - evaluate!(zero(x), result, x)))
    plt = plot(x, normalized_y)
    plot!(x, evaluate!(zero(x), result, x))
    display(plt)
    return norm(normalized_y - evaluate!(zero(x), result, x)) < threshold
end


function get_phase_model_with_phase_names(phase_names::AbstractSet,
                                          phases::AbstractArray,
                                          background::BackgroundModel=nothing,
                                          wildcard::OptionalPhases=nothing)

    l = []
    if isempty(phase_names)
        return PhaseModel(phases[l], wildcard, background)
    end

    for phase_name in phase_names
        for i in eachindex(phases)
            if phases[i].name == phase_name
                push!(l, i)
            end
        end
    end

    PhaseModel(phases[l], wildcard, background)
end

# returns a linear map from [a, b] to [0, 1]
unit_transform(a, b) = x->(x-a)/(b-a)
inv_unit_transform(a, b) = x -> (b-a)*x + a # inverse unit transform

