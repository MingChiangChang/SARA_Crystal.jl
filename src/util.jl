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
    println("Constructed background model")
    amorphous = PhaseModel(nothing, nothing, bg)
    println("Phase model constructed")
    println(typeof(amorphous))
    println(typeof(normalized_y))
    println(typeof(std_noise))
    println(typeof(mean_θ))
    println(typeof(maxiter))
    result = optimize!(amorphous, x, normalized_y, std_noise, mean_θ, std_θ, method=LM, objective="LS",
                       maxiter=maxiter, optimize_mode=Simple, regularization=true, verbose=true)
    println("optimized")
    println(norm(normalized_y - evaluate!(zero(x), result, x)))
    # plt = plot(x, normalized_y)
    # plot!(x, evaluate!(zero(x), result, x))
    # display(plt)
    return norm(normalized_y - evaluate!(zero(x), result, x)) < threshold
end

function get_phase_fractions!(fractions::AbstractMatrix,
                                W::AbstractMatrix,
                                H::AbstractMatrix,
                                result_nodes::AbstractVector,
                                h_thresh::Real,
                                frac_threshold::Real)
    W_max = [maximum(W[:, i]) for i in 1:size(W, 2)]
    for colindex in 1:size(H, 2)
        for i in eachindex(result_nodes)
            if  isassigned(result_nodes, i) && H[i, colindex] > h_thresh
                fracs = get_fraction(result_nodes[i].phase_model.CPs)
                for phase_idx in eachindex(fracs)
                    if fracs[phase_idx] >= frac_threshold # Threshold can be 0
                        fractions[colindex, result_nodes[i].phase_model.CPs[phase_idx].id] +=  W_max[i] * H[i, colindex] * fracs[phase_idx]
                    end
                end
            end
        end
    end
    fractions
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