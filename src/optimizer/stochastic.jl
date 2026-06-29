############################################################################################
## Stochastic mini-batch optimizer
############################################################################################

mutable struct SemOptimizerStochastic{F} <: SemOptimizer{:Stochastic}
    final_optimizer::F
    batch_size::Int
    iterations::Int
    η::Float64
    β₁::Float64
    β₂::Float64
    ϵ::Float64
    rng::Random.AbstractRNG
    shuffle::Bool
    finalize::Bool
end

"""
    SemOptimizer(; engine = :Stochastic, kwargs...)

ADAM-based stochastic mini-batch optimizer for case-wise SEM likelihoods.

The optimizer samples rows from the model's observed data, evaluates the SEM objective and
analytic gradient on each mini-batch, and updates parameters with ADAM. By default, the
stochastic estimates are then used as starting values for a final full-data likelihood
optimization step using `final_optimizer`.

This optimizer currently supports single-term `Sem` models whose observed object stores
case-wise data, such as `SemObservedMissing` with `SemFIML`.

# Keywords
- `batch_size = 32`: number of cases per mini-batch.
- `iterations = 1_000`: number of ADAM updates.
- `η = 0.001`: ADAM learning rate.
- `β₁ = 0.9`, `β₂ = 0.999`, `ϵ = 1e-8`: ADAM hyperparameters.
- `rng = Random.default_rng()`: random number generator used for row sampling.
- `shuffle = true`: sample without replacement within epochs when possible.
- `finalize = true`: run a final full-data optimization after ADAM.
- `final_optimizer = SemOptimizer(engine = :Optim)`: optimizer for the final full-data step.
"""
function SemOptimizerStochastic(;
    final_optimizer = SemOptimizer(engine = :Optim),
    batch_size::Integer = 32,
    iterations::Integer = 1_000,
    η::Real = 0.001,
    β₁::Real = 0.9,
    β₂::Real = 0.999,
    ϵ::Real = 1e-8,
    rng::Random.AbstractRNG = Random.default_rng(),
    shuffle::Bool = true,
    finalize::Bool = true,
    kwargs...,
)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    iterations >= 0 || throw(ArgumentError("iterations must be non-negative"))
    0 <= β₁ < 1 || throw(ArgumentError("β₁ must be in [0, 1)"))
    0 <= β₂ < 1 || throw(ArgumentError("β₂ must be in [0, 1)"))
    η > 0 || throw(ArgumentError("η must be positive"))
    ϵ > 0 || throw(ArgumentError("ϵ must be positive"))
    return SemOptimizerStochastic(
        final_optimizer,
        Int(batch_size),
        Int(iterations),
        Float64(η),
        Float64(β₁),
        Float64(β₂),
        Float64(ϵ),
        rng,
        shuffle,
        finalize,
    )
end

sem_optimizer_subtype(::Val{:Stochastic}) = SemOptimizerStochastic

struct SemStochasticResult{O <: SemOptimizerStochastic, R} <: SemOptimizerResult{O}
    optimizer::O
    stochastic_minimum::Float64
    stochastic_solution::Vector{Float64}
    objectives::Vector{Float64}
    final_result::R
end

n_iterations(res::SemStochasticResult) = length(res.objectives)
convergence(res::SemStochasticResult) = isnothing(res.final_result) ? Symbol[:iterations] : convergence(res.final_result)
converged(res::SemStochasticResult) = isnothing(res.final_result) ? true : converged(res.final_result)

function _casewise_data(model::Sem)
    nsem_terms(model) == 1 || throw(ArgumentError("Stochastic optimization currently supports single-term SEM models."))
    obs = observed(sem_term(model))
    hasproperty(obs, :data) || throw(ArgumentError("Stochastic optimization requires observed case-wise data."))
    return getproperty(obs, :data)
end

function _minibatch_indices!(batch, order, pos, n, optimizer::SemOptimizerStochastic)
    b = length(batch)
    if optimizer.shuffle && b <= n
        if pos[] + b - 1 > n
            randperm!(optimizer.rng, order)
            pos[] = 1
        end
        copyto!(batch, 1, order, pos[], b)
        pos[] += b
    else
        rand!(optimizer.rng, batch, 1:n)
    end
    return batch
end

function fit(
    optim::SemOptimizerStochastic,
    model::Sem,
    start_params::AbstractVector;
    lower_bounds::Union{AbstractVector, AbstractDict, Nothing} = nothing,
    upper_bounds::Union{AbstractVector, AbstractDict, Nothing} = nothing,
    lower_bound = -Inf,
    upper_bound = Inf,
    variance_lower_bound::Number = -Inf,
    variance_upper_bound::Number = Inf,
    kwargs...,
)
    data = _casewise_data(model)
    n = size(data, 1)
    n > 0 || throw(ArgumentError("Stochastic optimization requires at least one case."))
    batch_size = min(optim.batch_size, n)
    batch = Vector{Int}(undef, batch_size)
    order = collect(1:n)
    randperm!(optim.rng, order)
    pos = Ref(1)

    lbounds = prepare_param_bounds(
        Val(:lower),
        lower_bounds,
        model,
        default = lower_bound,
        variance_default = variance_lower_bound,
    )
    ubounds = prepare_param_bounds(
        Val(:upper),
        upper_bounds,
        model,
        default = upper_bound,
        variance_default = variance_upper_bound,
    )

    θ = clamp.(Float64.(start_params), lbounds, ubounds)
    m = zeros(length(θ))
    v = zeros(length(θ))
    g = zeros(length(θ))
    objectives = Float64[]

    for t in 1:optim.iterations
        inds = _minibatch_indices!(batch, order, pos, n, optim)
        batch_model = replace_observed(model, @view(data[inds, :]))
        obj = objective_gradient!(g, batch_model, θ)
        push!(objectives, Float64(obj))

        @. m = optim.β₁ * m + (1 - optim.β₁) * g
        @. v = optim.β₂ * v + (1 - optim.β₂) * g^2
        mhat = @. m / (1 - optim.β₁^t)
        vhat = @. v / (1 - optim.β₂^t)
        @. θ = θ - optim.η * mhat / (sqrt(vhat) + optim.ϵ)
        @. θ = min(max(θ, lbounds), ubounds)
    end

    stochastic_minimum = objective!(model, θ)
    final_result = optim.finalize ?
                   fit(
        optim.final_optimizer,
        model,
        θ;
        lower_bounds,
        upper_bounds,
        lower_bound,
        upper_bound,
        variance_lower_bound,
        variance_upper_bound,
        kwargs...,
    ) : nothing

    if optim.finalize
        return SemFit(
            minimum(final_result),
            solution(final_result),
            start_params,
            model,
            SemStochasticResult(optim, stochastic_minimum, θ, objectives, optimization_result(final_result)),
        )
    else
        return SemFit(
            stochastic_minimum,
            θ,
            start_params,
            model,
            SemStochasticResult(optim, stochastic_minimum, θ, objectives, nothing),
        )
    end
end

Base.show(io::IO, result::SemStochasticResult) = print(
    io,
    "ADAM stochastic mini-batch optimization ($(n_iterations(result)) iterations)",
)
