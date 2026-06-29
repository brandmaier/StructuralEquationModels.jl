using Test
using Random
using Optim
using StructuralEquationModels

@testset "stochastic ADAM optimizer for FIML" begin
    rng = MersenneTwister(20240629)
    observed_vars = Symbol.(:x, 1:4)
    latent_vars = [:f]

    graph = @StenoGraph begin
        f → fixed(1.0) * x1 + x2 + x3 + x4
        _(observed_vars) ↔ _(observed_vars)
        f ↔ f
        Symbol(1) → _(observed_vars)
    end

    partable = ParameterTable(graph; observed_vars, latent_vars)
    generating_model = Sem(; specification = partable, meanstructure = true)
    start = start_simple(generating_model)
    true_params = copy(start)
    true_params[1:3] .= [0.8, 1.1, 0.9]
    true_params[4:8] .= [0.35, 0.45, 0.4, 0.5, 1.0]
    true_params[9:12] .= [0.1, -0.1, 0.2, -0.2]

    data = rand(generating_model, true_params, 160)
    missing_data = Matrix{Union{Float64, Missing}}(data)
    for i in eachindex(missing_data)
        rand(rng) < 0.12 && (missing_data[i] = missing)
    end

    model = Sem(
        ;
        specification = partable,
        data = missing_data,
        observed = SemObservedMissing,
        loss = SemFIML,
        meanstructure = true,
    )

    optimizer = SemOptimizer(
        ;
        engine = :Stochastic,
        batch_size = 40,
        iterations = 8,
        η = 0.005,
        rng = MersenneTwister(99),
        final_optimizer = SemOptimizer(
            ;
            engine = :Optim,
            algorithm = Fminbox(LBFGS()),
            options = Optim.Options(; iterations = 20),
        ),
    )

    fit_stochastic = fit(
        optimizer,
        model;
        start_val = start,
        variance_lower_bound = 1e-4,
    )

    result = StructuralEquationModels.optimization_result(fit_stochastic)
    @test optimizer_engine(fit_stochastic) == :Stochastic
    @test result.final_result isa StructuralEquationModels.SemOptimResult
    @test length(result.objectives) == 8
    @test all(isfinite, result.objectives)
    @test all(isfinite, solution(fit_stochastic))
    @test isfinite(StructuralEquationModels.minimum(fit_stochastic))
end
