using Pkg

ssh = false
if ssh
    git = "git@github.com:SebastianAment/"
else
    git = "https://github.com/SebastianAment/"
end

add(x) = Pkg.add(Pkg.PackageSpec(url = git * x * ".git"))

add("LazyInverse.jl")
add("WoodburyIdentity.jl")
add("LinearAlgebraExtensions.jl")
add("LazyLinearAlgebra.jl")

add("KroneckerProducts.jl")

add("CovarianceFunctions.jl")
add("OptimizationAlgorithms.jl")
# add("BackgroundSubtraction.jl")
println("Trying to add GaussianDistributions.jl via ssh. If this fails, please add the package locally by typing \"]add path/to/GaussianDistributions.jl\" in the REPL.")
add("GaussianDistributions.jl")