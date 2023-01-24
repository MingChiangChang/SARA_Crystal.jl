struct StripeToGlobalSettings
    nmf_rank::Integer
    h_threshold::Real
    frac_threshold::Real
    background_length::Real
    kernel::AbstractKernel
    σ::Real
    TP::TemperatureProfile
    condition::NTuple
    input_noise::Union{Val{true}, Val{false}}
end

const STGSettings = StripeToGlobalSettings

function STGSettings()
    STGSettings(
        4, # nmf_rank
        0.1, # h_threshold
        0.1, # frac_threshold
        8.,  # background_length
        Lengthscale(MaternP(2), 0.05), # kernel
        0.05, # σ
        TemperatureProfile(), # TemperatureProfile
        (1300, 3), # condition
        Val(false)
    )
end

function STGSettings(nmf_rank::Integer, h_threshold::Real, frac_threshold::Real,
      background_length::Real, kernel::AbstractKernel, σ::Real, TP::TemperatureProfile,
      condition::NTuple)
      STGSettings(nmf_rank, h_threshold, frac_threshold, background_length, kernel,σ, TP,condition, Val{true})
end