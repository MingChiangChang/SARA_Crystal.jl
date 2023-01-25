function entropy_renormalize!(y::AbstractMatrix)
    for i in 1:size(y, 1)
        if(any(x -> x>0, y[i,:]))
            y[i,:] ./= sum(y[i, :])
        else
            y[i, :] .= 1/size(y, 2)
        end
    end
end

function renormalize!(y::AbstractMatrix)
    for i in 1:size(y, 1)
        if(any(x -> x>0, y[i,:]))
            y[i,:] ./= sum(y[i, :])
        end
    end
    y
end

get_entropy(p::Real) = p == 0 ? 0 : - p * log(p)
get_entropy(P::AbstractArray) = sum(get_entropy.(P))
get_entropy(P::AbstractMatrix) = reduce(vcat, sum(get_entropy.(P), dims=2))


####### Helper functions
const DEFAULT_RES_THRESH = 1.
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
    # println(norm(normalized_y - evaluate!(zero(x), result, x)))
    # plt = plot(x, normalized_y)
    # plot!(x, evaluate!(zero(x), result, x))
    # display(plt)
    return norm(normalized_y - evaluate!(zero(x), result, x)) < threshold
end

function scaling_fit(base_pattern::AbstractVector,
                     new_pattern::AbstractVector,
                     x0::AbstractVector)
    stn = LevenbergMarquartSettings(min_resnorm = 1e-2, min_res = 1e-10,
                    min_decrease = 1e-6, min_iter=10, max_iter = 128,
                    decrease_factor = 7, increase_factor = 10, max_step = 0.1)

    function t(r, p)
        @. r = new_pattern - base_pattern*p[1]
        r
    end

    lm = LevenbergMarquart(t, x0, zero(new_pattern))
    params, _ = OptimizationAlgorithms.optimize!(lm, x0, zero(new_pattern), stn, 1e-6, Val(false))
    return params
end

function get_phase_fractions!(fractions::AbstractMatrix,
                                W::AbstractMatrix,
                                H::AbstractMatrix,
                                amorphous_frac::AbstractVector,
                                result_nodes::AbstractVector,
                                h_thresh::Real,
                                frac_threshold::Real)
    # Collect amorphous contribution
    for i in axes(H, 1)
        for j in axes(H, 2)
            fractions[j, end] += H[i, j] * amorphous_frac[i]
        end
    end

    # Collect phase contribution
    W_max = [maximum(W[:, i]) for i in axes(W, 2)]
    for colindex in axes(H, 2)
        for i in eachindex(result_nodes)
            if  isassigned(result_nodes, i) && H[i, colindex] > h_thresh
                fracs = get_fraction(result_nodes[i].phase_model.CPs)
                for phase_idx in eachindex(fracs)
                    if fracs[phase_idx] >= frac_threshold # Threshold can be 0
                        fractions[colindex, result_nodes[i].phase_model.CPs[phase_idx].id+1] +=  W_max[i] * H[i, colindex] * fracs[phase_idx]
                    end
                end
            end
        end
    end

    #normalize
    for row in eachrow(fractions)
        if sum(row[1:end-1]) > 0.
            row[1:end-1] ./= sum(row[1:end-1]) / (1-row[end])
        else
            row[end] = 1.
        end
    end
    fractions
end

function classify_amorphous(W::AbstractMatrix, H::AbstractMatrix, n = 16)
    # Adapted from https://github.com/SebastianAment/PhaseMapping.jl
    h = sum(@view(H[:,1:n]), dims = 2)
    i = argmax(vec(h))
    return i, @view W[:, i]
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