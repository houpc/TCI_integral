using MCIntegration
include("integrand_FeynDiag.jl")
include("parameters_UEG.jl")

neval = 10^6

function integrand(x, config)
    return integrand_eval(x, config.userdata)
end

config = Configuration(; dof=[[var_dim]], var=Continuous(0, 1), userdata=configuration(rs, beta, order, mass2, Kdim))
result = integrate(integrand; config=config, neval=neval, print=0)

println("Monte Carlo Integration Result: $(result.mean[1]) +/- $(result.stdev[1])")