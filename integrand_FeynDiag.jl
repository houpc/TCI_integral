using ElectronLiquid, FeynmanDiagram
using DataFrames, JLD2, CSV
using LinearAlgebra

name = "sigma"
root_dir = joinpath(@__DIR__, "source_codeParquetAD/")
# include(root_dir * "Cwrapper_sigma_ParquetAD.jl")
# include("source_codeParquetAD/func_sigma_ParquetAD.jl")

# const evalfuncParquetAD_sigma_map = Dict(
#     (1, 0, 0) => eval_sigma_ParquetAD100!,
#     (2, 0, 0) => eval_sigma_ParquetAD200!,
#     (3, 0, 0) => eval_sigma_ParquetAD300!,
#     (4, 0, 0) => eval_sigma_ParquetAD400!,
#     (5, 0, 0) => eval_sigma_ParquetAD500!,
#     (6, 0, 0) => eval_sigma_ParquetAD600!
# )

struct LeafStateAD
    type::Int
    orders::Vector{Int}
    inTau_idx::Int
    outTau_idx::Int
    loop_idx::Int

    function LeafStateAD(type::Int, orders::Vector{Int}, inTau_idx::Int, outTau_idx::Int, loop_idx::Int)
        return new(type, orders, inTau_idx, outTau_idx, loop_idx)
    end
end

function configuration(rs, beta, order, mass2, dim=3)
    para = ParaMC(dim=dim, rs=rs, beta=beta, Fs=0.0, order=order, mass2=mass2, isDynamic=false)
    dim, kF = para.dim, para.kF
    key_str = join(string.((order, 0, 0)))

    extT_label = jldopen(root_dir * "extvars_$name.jld2")[key_str][1]

    maxMomNum = order + 1
    df = CSV.read(root_dir * "loopBasis_$(name)_maxOrder6.csv", DataFrame)
    loopBasis = [df[!, col][1:maxMomNum] for col in names(df)]
    momLoopPool = FrontEnds.LoopPool(:K, dim, loopBasis)

    df = CSV.read(root_dir * "leafinfo_$(name)_$key_str.csv", DataFrame)
    leafstate = Vector{LeafStateAD}()
    for row in eachrow(df)
        push!(leafstate, LeafStateAD(row[2], _StringtoIntVector(row[3]), row[4:end]...))
    end
    leafval = df[!, names(df)[1]]

    root = zeros(Float64, length(extT_label))
    ext_kgrid = [kF,]
    ext_ngrid = [0,]
    var_Kdata = zeros(dim, order + 1)
    var_Kdata[1, 1] = ext_kgrid[1]
    var_Tdata = zeros(order)

    return (para, var_Kdata, var_Tdata, ext_kgrid, ext_ngrid, extT_label, leafstate, leafval, momLoopPool, root)
end

@inline function _StringtoIntVector(str::AbstractString)
    pattern = r"[-+]?\d+"
    return [parse(Int, m.match) for m in eachmatch(pattern, str)]
end

function integrand_eval(vars, config)
    # para, var_Kdata, var_Tdata, ext_kgrid, ext_ngrid, extT_label, lfstat, leafval, momLoopPool, root = config.userdata
    para, var_Kdata, var_Tdata, ext_kgrid, ext_ngrid, extT_label, lfstat, leafval, momLoopPool, root = config
    dim, order, β, me, kF, λ, μ, e0, ϵ0 = para.dim, para.order, para.β, para.me, para.kF, para.mass2, para.μ, para.e0, para.ϵ0
    maxK = 10kF

    p_rescale = vars[1:order]
    theta = vars[order+1:2*order] * π
    phi = vars[2*order+1:3*order] * 2π

    var_Kdata[1, 1] = ext_kgrid[1]
    var_Kdata[1, 2:end] = p_rescale .* sin.(theta) * maxK
    var_Kdata[2, 2:end] = var_Kdata[1, 2:end] .* sin.(phi)
    var_Kdata[1, 2:end] .*= cos.(phi)
    var_Kdata[3, 2:end] = p_rescale .* cos.(theta) * maxK
    if order > 1
        var_Tdata[2:end] = vars[3*order+1:4*order-1] * β
    end

    FrontEnds.update(momLoopPool, var_Kdata)
    for (i, lfstat) in enumerate(lfstat)
        lftype, lforders, leafτ_i, leafτ_o, leafMomIdx = lfstat.type, lfstat.orders, lfstat.inTau_idx, lfstat.outTau_idx, lfstat.loop_idx
        if lftype == 0
            continue
        elseif lftype == 1 #fermionic 
            τ = var_Tdata[leafτ_o] - var_Tdata[leafτ_i]
            kq = FrontEnds.loop(momLoopPool, leafMomIdx)
            ϵ = dot(kq, kq) / (2me) - μ
            leafval[i] = ElectronLiquid.Propagator.green_derive(τ, ϵ, β, lforders[1])
        elseif lftype == 2 #bosonic 
            kq = FrontEnds.loop(momLoopPool, leafMomIdx)
            τ2, τ1 = var_Tdata[leafτ_o], var_Tdata[leafτ_i]
            leafval[i] = ElectronLiquid.Propagator.interaction_derive(τ1, τ2, kq, para, lforders; idtype=FrontEnds.Instant, tau_num=1)
            # if dim == 3
            #     invK = 1.0 / (dot(kq, kq) + λ)
            #     leafval[i] = e0^2 / ϵ0 * invK * (λ * invK)^order
            # elseif dim == 2
            #     invK = 1.0 / (sqrt(dot(kq, kq)) + λ)
            #     leafval[i] = e0^2 / 2ϵ0 * invK * (λ * invK)^order
            # end
        else
            error("this leaftype $lftype not implemented!")
        end
    end

    group = (para.order, 0, 0)
    Sigma.evalfuncParquetAD_sigma_map[group](root, leafval)

    factor = prod((p_rescale * maxK) .^ 2 .* sin.(theta))
    factor *= (2maxK * π^2)^order * β^(order - 1)
    factor /= (2π)^(dim * order)

    # n = ext_ngrid[1]
    # weight = sum(root[i] * phase(var_Tdata, extT, n, β) for (i, extT) in enumerate(extT_label))
    weight = sum(root)

    return weight * factor
end

@inline function phase(varT, extT, l, β)
    tin, tout = varT[extT[1]], varT[extT[2]]
    return exp(1im * π * (2l + 1) / β * (tout - tin))
end


## Test
# order = 1
# config = configuration(2.0, 32.0, order, 0.5)
# vars = rand(4 * order - 1)
# vars = [config[1].kF * 0.05, π / 2, π]
# println(integrand_eval(vars, config))