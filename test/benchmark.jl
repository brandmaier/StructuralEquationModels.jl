using StructuralEquationModels

graph = @StenoGraph begin
    # loadings
    T → fixed(1)*t1 + fixed(1)*t2 + fixed(1)*t3
    S12 → fixed(1)*t1 + fixed(1)*t2
    S3 → fixed(1)*t3
    # observed variances
    t1 ↔ label(:e)*t1
    t2 ↔ label(:e)*t2
    t3 ↔ label(:e)*t3
    # latent variances
    T ↔ T
    S12 ↔ label(:s)*S12
    S3 ↔ label(:s)*S3
end

spec = ParameterTable(graph = graph, observed_vars = [:t1, :t2, :t3], latent_vars = [:T, :S12, :S3])

obs = SemObservedCovariance(specification = nothing, obs_cov = rand(3,3))

model_true = Sem(;
    specification = spec,
    observed = obs
    )

model_true_sym = Sem(;
    specification = spec,
    observed = obs,
    imply = RAMSymbolic
    )

θₜ = [0.06, 0.86, 0.08]

objective!(model_true, θₜ)

import StructuralEquationModels: Σ

Σ_true = Σ(imply(model_true))

using Distributions, Random

data_vec = [permutedims(rand(MvNormal(Σ_true), 20)) for _ in 1:100]

#= function get_observed(; Σ_true, n_obs)
    data = permutedims(rand(MvNormal(Σ_true), n_obs))
    return data
end

function sem_fit_models(; get_observed, construct_model, kwargs...)
end

function fit_model(; Σ_true, n_obs)
    model = ...
end =#

### algorithm comparison

using Optim, LineSearches, BenchmarkTools, DataFrames

import StructuralEquationModels: convergence

algorithms = [LBFGS, BFGS]
linesearches = [HagerZhang(), MoreThuente(), BackTracking(), StrongWolfe(), Static()]
alphaguesses = [InitialPrevious(), InitialStatic(), InitialHagerZhang(), InitialQuadratic(), InitialConstantChange()]

strip_typepars(model::Sem) = Sem
strip_typepars(model::SemFiniteDiff) = SemFiniteDiff

function swap_algorithm(model, config; options = nothing)
    !isnothing(options) || (options = optimizer(model).options)
    opt = SemOptimizerOptim(algorithm = config, options = options)
    model = strip_typepars(model)(observed(model), imply(model), loss(model), opt)
    return model
end

function benchmark_optim_config(
    model;
    configs = [
        LBFGS(; linesearch = HagerZhang()),
        BFGS(; linesearch = HagerZhang()),
        LBFGS(; linesearch = MoreThuente()),
        BFGS(; linesearch = MoreThuente()),
        LBFGS(; linesearch = BackTracking()),
        BFGS(; linesearch = BackTracking()),
    ],
    kwargs...
    )

    converged = []
    mean_times = []

    for config in configs
        model = swap_algorithm(model, config)
        fit = sem_fit(model; kwargs...)
        push!(converged, convergence(fit))
        bench = @benchmark sem_fit(model)
        push!(mean_times, mean(bench.times))
    end

    return DataFrame(:config => configs, :mean_time => mean_times, :convergence => converged)

end

model = swap_observed(model_true, SemObservedData(data = data_vec[2], specification = nothing))

model_sym = swap_observed(
    model_true_sym, 
    SemObservedData(data = data_vec[2], specification = nothing),
    specification = spec)

using MKL

bm = benchmark_optim_config(model)

using Plots

bm.mean_time = bm.mean_time/1e6 # convert to ms

bm.algo = repeat(["LBFGS", "BFGS"], 3)
bm.ls = ["HZ", "HZ", "MT", "MT", "BT", "BT"]


plot(bm.ls, bm.mean_time; seriestype = :scatter, ylims = (0, Inf))

algorithms = [LBFGS, BFGS]
linesearches = [HagerZhang(), MoreThuente(), BackTracking(), StrongWolfe(), Static()]
alphaguesses = [InitialPrevious(), InitialStatic(), InitialHagerZhang(), InitialQuadratic(), InitialConstantChange()]


linesearches = [HagerZhang()]

configs = [algorithm(; alphaguess = alphaguess, linesearch = linesearch) 
    for algorithm in algorithms 
        for linesearch in linesearches
            for alphaguess in alphaguesses]

bm = benchmark_optim_config(model; configs = configs)

all(bm.convergence)

bm.mean_time = bm.mean_time/1e6 # convert to ms

plot(bm.mean_time, ylims = (0, Inf))

start = solution(sem_fit(model))

@benchmark sem_fit(model; start_val = start)

### multiple datasets

function fit_models(data_vec, model; kwargs...)
    new_obs = SemObservedData(data = data_vec[1], specification = nothing)
    new_model = swap_observed(model, new_obs; kwargs...)
    first_fit = sem_fit(new_model)

    solutions = [solution(first_fit)]
    minima = [StructuralEquationModels.minimum(first_fit)]

    for (i, data) in zip(2:length(data_vec), data_vec[2:end])
        new_obs = SemObservedData(data = data, specification = nothing)
        new_model = swap_observed(new_model, new_obs; kwargs...)
        new_fit = sem_fit(new_model; start_val = solutions[i-1])
        push!(solutions, solution(new_fit))
        push!(minima, StructuralEquationModels.minimum(new_fit))
    end

    return solutions, minima
end

using MKL, Plots

solutions, minima = fit_models(data_vec, model_true_sym)

sol_wide = hcat(solutions...)

plot(sol_wide')

@benchmark fit_models(data_vec, model_true)

@benchmark fit_models(data_vec, model_true_sym)

using PProf

function f(n)
    for _ in 1:n
        fit_models(data_vec, model_true_sym)
    end
end

@profile f(10)

pprof()

Profile.clear()