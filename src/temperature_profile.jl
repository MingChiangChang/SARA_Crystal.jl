abstract type AbstractTemperatureProfile end
struct GaussianTemperatureProfile{FWHM, U, V} <: AbstractTemperatureProfile
    # full-width half-max function for Gaussian temperature profile
    width::FWHM
    # parameters governing uncertainty calculation
    dT::U
    dx::V
end
const TemperatureProfile = GaussianTemperatureProfile

# Constructors:
# defaults to parameters of Science Advances submission
function TemperatureProfile(dT::Real = 20, dx::Real = 10/1000)
    get_temperature_profile_Bi2O3_2021(dT, dx)
end

# template for t-profile constructor
function get_temperature_profile_template(dT::Real = 20, dx::Real = 10/1000)
    println("temperature_profile_template")
    function width(T, log10_τ)
        throw("not implemented - should be a functional expression from temperature and dwell time to the fwhm of the profile")
    end
    TemperatureProfile(width, dT, dx)
end

# temperature profile for the Bi2O3 data in the Science Advances submission
function get_temperature_profile_Bi2O3_2021(dT::Real = 20, dx::Real = 10/1000)
    function width(T, log10_τ)
        fwhm_0 = 1.60742002e+03
        a = (-1.46900821e+00, -2.06060111e+02, 3.05703021e-01, 3.29193895e+02, -1.23361265e+13, 1.09165256e+01, -1.69547934e+04, -1.42207444e-04)
        fwhm_0 + a[1]*T + a[2]*log10_τ + a[3]*log10_τ*T + a[4]*log10_τ^4/T +
            a[5]/log10_τ^8/T^3 + a[6]*T/log10_τ^3 + a[7]/log10_τ^4 + a[8]*T^3/log10_τ^8
    end
    TemperatureProfile(width, dT, dx)
end

# OLD temperature profile for the Bi2O3 data
function get_temperature_profile_Bi2O3_2020(dT::Real = 20, dx::Real = 10/1000)
    function width(T, log10_τ)
        fwhm_0 = 6.21599199e+01
        a = (-3.56990024e-01, 2.84613876e+02, -8.90507983e-03, -5.76367089e+02, -1.95522786e+13, 5.51350589e+00, -3.34684841e+03, -1.33972469e-04)
        fwhm_0 + a[1]*T + a[2]*log10_τ + a[3]*log10_τ*T + a[4]*log10_τ^4/T +
            a[5]/log10_τ^8/T^3 + a[6]*T/log10_τ^3 + a[7]/log10_τ^4 + a[8]*T^3/log10_τ^8
    end
    TemperatureProfile(width, dT, dx)
end

# temperature profile for CHESS 2021
function get_temperature_profile_CHESS_Spring_2021(dT::Real = 20, dx::Real = 10/1000)
    # power = current in A
    # tau = dwell in us
    function _std(power, log10_τ)
        mm_per_pixel = 0.00153846153 # millimeters per pixel
        μm_per_pixel = mm_per_pixel * 1e3 # micrometers per pixel
        μm_per_pixel * (-6.0585 * power + 5.29285 * log10_τ + 1.432e-2 * power^2 - 1.59943e1 *log10_τ^2 + 7.6715e-1 * power*log10_τ + 726.0242)
    end
    _power(T_max::Int, log10_τ) = _power(float(T_max), float(log10_τ))
    _power(T_max, log10_τ) = (5000*sqrt(5)*sqrt(614000*T_max+1327591125*log10_τ^2 -8080505000*log10_τ+12142813538)-341554591*log10_τ+1065621710)/3070000
    _fwhm(power, log10_τ) = _std(power, log10_τ) / 2 # by the definitions of variables in the Gaussian-like function Max used
    width(T_peak, log10_τ) = _fwhm(_power(T_peak, log10_τ), log10_τ) # getting the power from the power profile
    TemperatureProfile(width, dT, dx)
end

# for Chess 2021 fall
# power = current in A
# tau = dwell in us
function get_temperature_profile_CHESS_Fall_2021(dT::Real = 25, dx::Real = 50/1000)
    function _std(power, log10_τ)
        mm_per_pixel = 0.001577 # 0.00153846153  millimeters per pixel
        μm_per_pixel = mm_per_pixel * 1e3 # micrometers per pixel
        μm_per_pixel * (6.70640332e+02 -2.22189018e+02*log10_τ -2.27743561e+00*power + 3.02469331e+01*log10_τ^2 + 7.70978201e-03*power^2 +1.70903683e-01*log10_τ*power)
    end
    _fwhm(power, log10_τ) = _std(power, log10_τ) / 2 # by the definitions of variables in the Gaussian-like function Max used
    _power(T_max::Int, log10_τ) = _power(float(T_max), float(log10_τ))
    _power(T_max, log10_τ) = (-6.40883896736885e+15*log10_τ^2 + 3.78789311741561e+16*log10_τ + 4.64073240375504e+16*sqrt(0.0190715357103267*log10_τ^4 - 0.180649351400508*log10_τ^3 + 0.651605603621984*log10_τ^2 + 5.21414120166499e-5*log10_τ*T_max - log10_τ - 0.000206391126900198*T_max + 0.569129395966527) - 7.48733837342515e+16)/(112293816201517.0*log10_τ - 444492129640718.0)
    width(T_peak, log10_τ) = _fwhm(_power(T_peak, log10_τ), log10_τ) # getting the power from the power profile
    TemperatureProfile(width, dT, dx)
end

# Functor definitions:
# WARNING: temperature profile is a function of position in mm
function (P::TemperatureProfile)(T_max::Real, log10_τ::Real)
    f = P.width(T_max, log10_τ)
    x->_T_helper(T_max, f, 1e3*x) # multiplying position by 1e3 to convert from mm to μm
end
function (P::TemperatureProfile)(T_max::Real, log10_τ::Real, x::Real)
    P(T_max, log10_τ)(x)
end
# WARNING: temperature profile is a function of position in μm
# function _T_helper(T_max, fwhm, ampl, x::Real)
#     return @. T_max * exp(-abs(2x / fwhm)^ampl)
# end
function _T_helper(T_max, fwhm, x::Real)
    return @. T_max * exp(-(2x / fwhm)^2)
end

profile(P::AbstractTemperatureProfile, T_max, log10_τ) = P(T_max, log10_τ)
# IDEA: generalize to non-analytically invertable T profiles using Newton's method
# returns the positive side of the inverse mapping T to position in mm
function inverse_profile(P::TemperatureProfile, T_max::Real, log10_τ::Real)
    fwhm = P.width(T_max, log10_τ)
    function invprof(T)
        T > 0 || throw(DomainError("T is non-positive: $T"))
        T_max ≥ T || throw(DomainError("T exceeds T_max: T = $T > $T_max = T_max"))
        (T ≈ T_max) ? zero(eltype(T)) : sqrt(log(T_max) - log(T)) * fwhm/2 * (1e-3)
    end
end

# x is a position
# computes uncertainty of the temperature profile
# Two components:
# - multiplicative and
# - proportional to derivative
function temperature_uncertainty(P::AbstractTemperatureProfile, T_max, log10_τ,
    dT::Real = P.dT, dx::Real = P.dx; maximum_allowed_T::Real = 1400)
    f = P(T_max, log10_τ)
    function temperature_variance(x::Real)
        T, dTdx = f(x), ForwardDiff.derivative(f, x)
        (dT * T / maximum_allowed_T)^2 + (dx * dTdx)^2 # error variance is additive
    end
end


# more generic temperature inversion function based on Newton algorithm
function inverse_profile(profile, T_max::Real, log10_τ::Real;
                    newton_iter::Int = 4, maximum_x::Real = 3) # maximum x from midpoint in mm
    f(x::Real) = profile(T_max, log10_τ, x)
    df(x::Real) = ForwardDiff.derivative(f, x)
    function invprof(T) # solving for x in profile(T_max, log10_τ, x) = T
        T > 0 || throw(DomainError("T is non-positive: $T"))
        T_max ≥ T || throw(DomainError("T exceeds T_max: T = $T > $T_max = T_max"))
        if T ≈ T_max
            zero(eltype(T))
        else
            g(x) = f(x) - T # find root of this function to have profile(T_max, log10_τ, x) = T
            x = find_zero(g, (0, maximum_x), Bisection(), atol = 1) # locate root approximately
            for _ in 1:newton_iter # followed by a few Newton iterations
                x -= g(x) / df(x)
                if x < 0 # if we ever diverge to the negative side, just return using Bisection algorithm
                    return find_zero(g, (0, maximum_x), Bisection())
                end
            end
            return x
        end
    end
end

################################################################################
# remove all data below one degree, since these are not interesting and can cause numerical issues
# assumes that Y is spectrogroscopic variable (i.e. q or wavelength λ) by position
function remove_low_temperature_data(x::AbstractVector, Y::AbstractMatrix,
            P::AbstractTemperatureProfile, T_max::Real, log10_τ::Real, min_T::Real = 1)
    T = profile(P, T_max, log10_τ).(x) # maps position of measured optical data to temperature
    i = T .> min_T
    x = x[i]
    Y = Y[:, i]
    return x, Y
end

# y is vector of vectors of position
function remove_low_temperature_data(x::AbstractVector, y::AbstractVector{<:AbstractVector},
            P::AbstractTemperatureProfile, T_max::Real, log10_τ::Real, min_T::Real = 1)
    T = profile(P, T_max, log10_τ).(x) # maps position of measured optical data to temperature
    i = T .> min_T
    x = x[i]
    y = [yj[i] for yj in y]
    return x, Y
end

########################### CHESS April 2022 ###################################
struct LorentzianTemperatureProfile{WT, U, V} <: AbstractTemperatureProfile
    width::WT
    dT::U
    dx::V
end

function get_temperature_profile_CHESS_Spring_2022(dT::Real = 20, dx::Real = 10/1000)
    pfit = [0.00903015256, -0.000174216105, 0.0742548071, -0.555849208, 3.5668932]
    left_width_fit  = [281.92169684,  -9.4047936 , -2.19357261]
    right_width_fit = [211.76451937,  12.60267879, -2.53188879]
    width_fit = left_width_fit
    inv_T_power = chess22_inverse_temp_surface(pfit)
    width_fun_pixel = chess22_width_of_left_lorentzian(width_fit)
    μm_per_pixel = 1.04
    function width_μm(T_max::Real, log10_τ::Real)
        # convert log10_τ -> velocity
        τ = 10^log10_τ
        log10_velocity = log10(88200 / τ)
        # get power
        power = inv_T_power(log10_velocity, T_max)
        # get width
        width_pixel = width_fun_pixel(log10_velocity, power)
        width_μm = μm_per_pixel * width_pixel
        return -width_μm # return negative of width to return negative positions (where T_profile is most accurate)
    end
    LorentzianTemperatureProfile(width_μm, dT, dx)
end

function (P::LorentzianTemperatureProfile)(T_max::Real, log10_τ::Real)
    (x_mm) -> P(T_max, log10_τ, x_mm)
end
function (P::LorentzianTemperatureProfile)(T_max::Real, log10_τ::Real, x_mm::Real)
    x_μm = 1e3 * x_mm
    width_μm = P.width(T_max, log10_τ)
    return T = T_max * inv(1 + (x_μm / width_μm)^2) # evaluate profile
end

function inverse_profile(P::LorentzianTemperatureProfile, T_max::Real, log10_τ::Real)
    width_μm = P.width(T_max, log10_τ)
    function invprof(T)
        T > 0 || throw(DomainError("T is non-positive: $T"))
        T_max ≥ T || throw(DomainError("T exceeds T_max: T = $T > $T_max = T_max"))
        # derivation of inverse
        # T = T_max * inv(1 + (x_μm / width_μm)^2)
        # inv(T / T_max) = 1 + (x_μm / width_μm)^2
        (T ≈ T_max) ? zero(eltype(T)) : sqrt(inv(T / T_max) - 1) * width_μm * 1e-3 # = x_mm (negative position for CHESS 2022, since width is negative)
    end
end

# converts (log_velocity, T_max) -> power
function chess22_inverse_temp_surface(pfit)
    b, c, d, e, f = pfit
    function inv_T_power(log10_velocity, T_max)
        (T_max / (b*log10_velocity+c))^(1/(d*log10_velocity^2 + e*log10_velocity +f))
    end
end

function chess22_width_of_left_lorentzian(width_fit)
    a, b, c = width_fit
    function (log_velo, power)
        a + b*log_velo + c*power
    end
end

### 2023
function get_temperature_profile_CHESS_Spring_2023(dT::Real = 20, dx::Real = 10/1000)
    # pfit = [0.00903015256, -0.000174216105, 0.0742548071, -0.555849208, 3.5668932]
    # pfit = [-0.06688143,   0.22353149,  -0.0556677,   0.20606041,   2.43702215]
    pfit = [-0.05415743,  0.19816566, -0.09018768,  0.28602151,  2.41121985]
    left_width_fit  = [ 7.93197988e+02,  2.34346130e+01, -4.29544003e+01,  4.68763599e+01,
                        9.64778097e-01, -4.95937074e+00, -1.68873411e+00, -9.11029166e-05]
    right_width_fit = [ 4.39339896e+02, -4.51657906e+01, -9.35886967e+00, -3.51825565e+00,
                        -2.20075111e-02,  2.44098710e+00]
    # width_fit = left_width_fit
    inv_T_power = chess23_inverse_temp_surface(pfit)
    left_width_func_pixel = chess23_width_of_left_lorentzian(left_width_fit)
    right_width_func_pixel = chess23_width_of_right_lorentzian(right_width_fit)
    μm_per_pixel = 1.08 # FIXME: measure the pixel size
    function width_μm(T_max::Real, log10_τ::Real)
        # convert log10_τ -> velocity
        τ = 10^log10_τ
        log10_velocity = log10(88200 / τ)
        # get power
        power = inv_T_power(log10_velocity, T_max)
        # get width
        width_pixel = right_width_func_pixel(log10_velocity, power)
        width_μm = μm_per_pixel * width_pixel
        return width_μm # return positive to use right width 
    end
    LorentzianTemperatureProfile(width_μm, dT, dx)
end

function inverse_profile(P::LorentzianTemperatureProfile, T_max::Real, log10_τ::Real)
    width_μm = P.width(T_max, log10_τ)
    function invprof(T)
        T > 0 || throw(DomainError("T is non-positive: $T"))
        T_max ≥ T || throw(DomainError("T exceeds T_max: T = $T > $T_max = T_max"))
        # derivation of inverse
        # T = T_max * inv(1 + (x_μm / width_μm)^2)
        # inv(T / T_max) = 1 + (x_μm / width_μm)^2
        (T ≈ T_max) ? zero(eltype(T)) : sqrt(inv(T / T_max) - 1) * width_μm * 1e-3 # = x_mm (negative position for CHESS 2022, since width is negative)
    end
end

# converts (log_velocity, T_max) -> power
function chess23_inverse_temp_surface(pfit)
    b, c, d, e, f = pfit
    function inv_T_power(log10_velocity, T_max)
        (T_max / (b*log10_velocity+c))^(1/(d*log10_velocity^2 + e*log10_velocity +f))
    end
end

function chess23_width_of_left_lorentzian(width_fit)
    base, a, b, c, d, e, f, g = width_fit
    function (log_velo, power)
        base + a*log_velo + b*power + c*log_velo^2 + d*power^2 + e*log_velo*power + f*log_velo^4 + g*power^4
    end
end


function chess23_width_of_right_lorentzian(width_fit)
    base, a, b, c, d, e = width_fit
    function (log_velo, power)
        base + a*log_velo + b*power + c*log_velo^2 + d*power^2 + e*log_velo*power
    end
end


#### Helper
function get_relevant_T(T_max::Real, log10_τ::Real, constant::Val{true},
        c_low::Real, c_high::Real, n::Int,)
    collect(range(T_max - c_low, T_max - c_high, length = n))
end

# relevant temperatures with proportional offset from T_max
function get_relevant_T(T_max::Real, log10_τ::Real, constant::Val{false},
    p_min::Real, p_max::Real, n::Int)
    collect(T_max * range(p_min, p_max, length = n))
end

# lazifies evaluation on T_max and log10_τ
function get_relevant_T(constant::Union{Val{true}, Val{false}}, c_min::Real, c_max::Real, n::Int)
    get_T(T_max::Real, log10_τ::Real) = get_relevant_T(T_max, log10_τ, constant, c_min, c_max, n)
end

# makes constant offset the default
function get_relevant_T(c_min::Real, c_max::Real, n::Int)
    constant = Val(true)
    get_T(T_max::Real, log10_τ::Real) = get_relevant_T(T_max, log10_τ, constant, c_min, c_max, n)
end