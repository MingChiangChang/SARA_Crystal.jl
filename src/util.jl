function entropy_renormalize!(y::AbstractMatrix)
    for i in 1:axes(y, 1)
        if(any(x -> x>0, y[i,:]))
            y[i,:] ./= sum(y[i, :])
        else
            y[i, :] .= 1/size(y, 2)
        end
    end
end

function renormalize!(y::AbstractMatrix)
    for i in axes(y, 1)
        if(any(x -> x>0, y[i,:]))
            y[i,:] ./= sum(y[i, :])
        end
    end
    y
end

get_entropy(p::Real) = p <= 0 ? zero(p) : - p * log(p)
get_entropy(P::AbstractArray) = sum(get_entropy.(P))
get_entropy(P::AbstractMatrix) = reduce(vcat, sum(get_entropy.(P), dims=2))


####### Helper functions
function reject_probs(sorted_probs::AbstractVector)
    # check probabilties for reject conditions
    if std(sorted_probs) < STD_THRESHOLD
        println("Rejected")
        println("Reason: Standard deviation of the top 5 node is not high enough")
        return true
    elseif sorted_probs[1] < TOP_PROB_THRESHOLD
        println("Rejected")
        println("Reason: Top probability not high enough")
        return true
    elseif (sorted_probs[1]-sorted_probs[5]) < TOP_FIVE_DIFF_THRESHOLD
        println("Rejected")
        println("Reason: Top five node difference is not big enough")
        return true
    end
    return false
end



function is_amorphous(x::AbstractVector, y::AbstractVector, l::Real, p::Real,
                    std_noise::Real=0.05,
                    mean_θ::AbstractVector=[1., 1., 1.],
                    std_θ::AbstractVector=[0.05, 0.5, 0.2],
                    maxiter::Int=512,
                    threshold::Real = DEFAULT_RES_THRESH)
    println("#######")
    println("Doing Amorphous Check!!!")
    println("#######")
    normalized_y =  y./maximum(y)
    bg = BackgroundModel(x, EQ(), l, p)
    amorphous = PhaseModel(nothing, nothing, bg)
    result = optimize!(amorphous, x, normalized_y, std_noise, mean_θ, std_θ, method=LM, objective="LS",
                       maxiter=maxiter, optimize_mode=Simple, regularization=true, verbose=false)
    # println(norm(normalized_y - evaluate!(zero(x), result, x)))
    # plt = plot(x, normalized_y)
    # plot!(x, evaluate!(zero(x), result, x))
    # display(plt)
    # IDEA: can do peak detection of background subtracted pattern instead of l2 norm
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

function calculate_unnormalized_fractions!(fractions::AbstractMatrix,
                                W::AbstractMatrix,
                                H::AbstractMatrix,
                                phase_fraction_of_bases::AbstractMatrix,
                                amorphous_frac::AbstractVector,
                                result_nodes::AbstractVector,
                                h_thresh::Real,
                                frac_thresh::Real)
    # Collect amorphous contribution
    collect_amorphous_frac!(fractions, H, amorphous_frac)
    collect_phase_contribution!(fractions, W, H, result_nodes, phase_fraction_of_bases, h_thresh, frac_thresh)
    getproperty.(fractions, :val), getproperty.(fractions, :err)
end

function collect_amorphous_frac!(fractions, H, amorphous_frac)
    for i in axes(H, 1)
        for j in axes(H, 2)
            fractions[j, end] += H[i, j] * amorphous_frac[i]
        end
    end
end

function collect_phase_contribution!(fractions, W, H, result_nodes, phase_fraction_of_bases, h_thresh, frac_thresh)
    W_max = [maximum(W[:, i]) for i in axes(W, 2)]
    for colindex in axes(H, 2)
        for i in axes(phase_fraction_of_bases, 1)
            if isassigned(result_nodes, i) && H[i, colindex] > h_thresh
                fraction_value = getproperty.(phase_fraction_of_bases[i,:], :val)
                fraction_value ./= maximum(fraction_value)
                for phase_idx in eachindex(fraction_value)
                    if fraction_value[phase_idx] >= frac_thresh # Threshold can be 0
                        fractions[colindex, phase_idx] +=  W_max[i] * H[i, colindex] * phase_fraction_of_bases[i, phase_idx]
                    end
                end
            end
        end
    end
end


function normalize_with_amorphous!(fractions)
    for row in eachrow(fractions)
        #println(sum(getproperty.(row[1:end-1], :val)))
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

function get_expected_entropy(Ws::AbstractMatrix,
                              Hs::AbstractMatrix,
                              amorphous_frac::AbstractVector,
                              nodes::AbstractMatrix,
                              probability::AbstractMatrix,
                              num_phase::Integer,
                              top_node_count::Int=DEFAULT_TOP_NODE_COUNT)

    node_ind = get_assigned_indices(nodes)
    entropy = zeros(Float64, size(Hs, 2))
    node_combinations, overall_prob = get_node_conbination_and_prob(nodes,
                                                                    node_ind,
                                                                    probability,
                                                                    top_node_count)
    normalization = [maximum(Ws[:, i]) for i in axes(Ws, 2)]

    for i in axes(Hs, 2)
        entropy[i] = weighted_entropy(overall_prob,
                combine_activations_to_fraction(normalization,
                                                Hs[:, i],
                                                amorphous_frac[i],
                                                node_combinations,
                                                num_phase))
    end

    entropy
end

function get_expected_fraction(Ws::AbstractMatrix,
                        Hs::AbstractMatrix,
                        amorphous_frac::AbstractVector,
                        nodes::AbstractMatrix,
                        probability::AbstractMatrix,
                        num_phase::Integer,
                        top_node_count::Int=DEFAULT_TOP_NODE_COUNT)

    node_ind = get_assigned_indices(nodes)

    fractions = zeros(Float64, (size(Hs, 2), num_phase+1))
    if isempty(node_ind)
        fractions[:,end] .= 1.0
    else
        node_combinations, overall_prob = get_node_conbination_and_prob(nodes,
                                                                        node_ind,
                                                                        probability,
                                                                        top_node_count)
        normalization = [maximum(Ws[:, i]) for i in axes(Ws, 2)]

        for i in axes(Hs, 2)
            fractions[i, :] = sum(overall_prob .* combine_activations_to_fraction(normalization,
                                                                            Hs[:, i],
                                                                            amorphous_frac[i],
                                                                            node_combinations,
                                                                            num_phase), dims=1)
        end
    end

    fractions
end

function get_expected_fraction_and_l2(Ws::AbstractMatrix,
                                        Hs::AbstractMatrix,
                                        amorphous_frac::AbstractVector,
                                        nodes::AbstractMatrix,
                                        q::AbstractVector,
                                        probability::AbstractMatrix,
                                        num_phase::Integer,
                                        top_node_count::Int=DEFAULT_TOP_NODE_COUNT)

    node_ind = get_assigned_indices(nodes)

    fractions = zeros(Float64, (size(Hs, 2), num_phase+1))
    fractions = zeros(Float64, (size(Hs, 2)))

    if isempty(node_ind)
        fractions[:,end] .= 1.0
    else
        node_combinations, overall_prob = get_node_conbination_and_prob(nodes,
                                                            node_ind,
                                                            probability,
                                                            top_node_count)
        normalization = [maximum(Ws[:, i]) for i in axes(Ws, 2)]

        for i in axes(Hs, 2)
            fractions[i, :] = sum(overall_prob .* combine_activations_to_fraction(normalization,
                                                                    Hs[:, i],
                                                                    amorphous_frac[i],
                                                                    node_combinations,
                                                                    num_phase), dims=1)
            l2[i, :] = sum(overall_prob .* residual(q, node_combinations))
        end
    end

    fractions
end

# Sum l2 residual for different NMF basis
# Maybe should do reoptimization
function residual(q::AbstractVector, Ws, Hs, node_combinations::AbstractVector)
    res = zeros(size(node_combinations, 1))

    for i in eachindex(res)
        for j in axes(Hs, 1)
            res[i] += Hs[j, i] * sum(abs2, Ws[:, i] - evaluate!(zero(q), node_combinations[i], q))
        end
    end

    res
end

function get_assigned_indices(node_matrix::AbstractMatrix)
    node_ind = Int64[]
    for i in axes(node_matrix, 1)
        if isassigned(node_matrix, i, 1)
            push!(node_ind, i)
        end
    end
    node_ind
end

function get_node_conbination_and_prob(nodes::AbstractMatrix,
                                      node_ind::AbstractVector,
                                      probability::AbstractMatrix,
                                      top_node_count::Int=DEFAULT_TOP_NODE_COUNT)

    arr = reduce(vcat, [repeat([i], length(node_ind)) for i in 1:top_node_count])
    all_perm = collect(multiset_permutations(arr, length(node_ind)))

    overall_prob = similar(all_perm, Float64)
    node_combinations = Array{Node, 2}(undef, (length(all_perm), length(node_ind)))

    for i in eachindex(all_perm)
        overall_prob[i] = prod(getindex.((probability,), node_ind, all_perm[i]))
        node_combinations[i, :] = getindex.((nodes,), node_ind, all_perm[i])
    end
    overall_prob ./= sum(overall_prob)

    node_combinations, overall_prob
end

""" Calculate the combined phase fraction at each location for each combination"""
function combine_activations_to_fraction(normalization::AbstractVector, H::AbstractVector, amorphous_frac::Real,
                                         node_combinations::AbstractMatrix, num_phase::Integer)

    phase_fraction = zeros(size(node_combinations, 1), num_phase+1)

    for combination_idx in axes(phase_fraction, 1)
        for i in eachindex(node_combinations[combination_idx, :])
            for CP in node_combinations[combination_idx, i].phase_model.CPs
                phase_fraction[combination_idx, CP.id+1] += normalization[i] * H[i] * CP.act / CP.norm_constant
            end
        end
    end

    phase_fraction[:, end] .= amorphous_frac
    normalize_with_amorphous!(phase_fraction)

    phase_fraction
end


function weighted_entropy(probabilities::AbstractVector, phase_fractions::AbstractMatrix)
    for i in axes(phase_fractions, 1)
        phase_fractions[i, 1:end-1] .+= phase_fractions[i, end]/(size(phase_fractions, 2) -1)
    end
    entropies = get_entropy(phase_fractions[:, 1:end-1])
    sum(probabilities .* entropies)
end

# returns a linear map from [a, b] to [0, 1]
unit_transform(a, b) = x->(x-a)/(b-a)
inv_unit_transform(a, b) = x -> (b-a)*x + a # inverse unit transform