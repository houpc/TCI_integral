using PythonCall: PythonCall
using PythonPlot: pyplot as plt, Figure
using PythonPlot
import QuanticsGrids as QG
using QuanticsTCI: quanticscrossinterpolate, integral
import TensorCrossInterpolation as TCI
using LaTeXStrings

PythonPlot.matplotlib.rcParams["font.size"] = 18

include("integrand_FeynDiag.jl") # integrand functions of self-energy diagrams
include("parameters_UEG.jl") # parameters of the uniform electron gas (UEG) model

MCresults = [0.2387, 0.22845, -0.03115]  # Monte Carlo results for rs=2, beta=32, mass2=0.5, Kdim=3, and order=[1,2,3]
MCresult = MCresults[order]
# MCresult = 0.184437  # beta=1, order=1

const config = configuration(rs, beta, order, mass2, Kdim)

function func(x)
    return integrand_eval(collect(x), config)
end

R = 30 # number of bits
xmin = 0.0
xmax = 1.0
N = 2^R # size of the grid
tol = 1e-4 # Tolerance for the error
maxDb = 200 # maximum bond dimension

qgrid = QG.DiscretizedGrid{var_dim}(R, Tuple(fill(xmin, var_dim)), Tuple(fill(xmax, var_dim)); unfoldingscheme=:interleaved, includeendpoint=true)
quanticsf(sigma) = func(QG.quantics_to_origcoord(qgrid, sigma))

# pivots to be used for initialization
init_pivots = Vector{Vector{Int}}()
pivot_point = zeros(var_dim)
println("Initial pivot points: ")
for k in [0.1]
    pivot_point[1:order] .= k
    for theta in [0.5]
        pivot_point[order+1:2*order] .= theta
        for phi in [0.1, 0.9]
            pivot_point[2*order+1:3*order] .= phi
            if order > 1
                for tau in [0.05, 0.95]
                    pivot_point[3*order+1:end] .= tau
                    println(pivot_point)
                    push!(init_pivots, QG.origcoord_to_quantics(qgrid, Tuple(pivot_point)))
                end
            else
                println(pivot_point)
                push!(init_pivots, QG.origcoord_to_quantics(qgrid, Tuple(pivot_point)))
            end
        end
    end
end

# Perform QTCI
ci, ranks, errors = TCI.crossinterpolate2(Float64, quanticsf, QG.localdimensions(qgrid), init_pivots;
    maxbonddim=maxDb, tolerance=tol,
    normalizeerror=true, # Normalize the error by the maximum sample value,
    verbosity=1, loginterval=1, # Log the error every `loginterval` iterations
)
bond_dims = TCI.linkdims(ci)
println("Bond dimensions: $bond_dims")
# println("pivot errors: $(ci.pivoterrors ./ ci.maxsamplevalue)")

integralvalue = TCI.sum(ci) * prod(QG.grid_step(qgrid))
println("QTCI Integral result: $integralvalue")
println("Monte Carlo Integral result: $MCresult")


# Plot the function and the TCI reconstructions for (τ_1 and) k_1 scan
grididx_pivot1 = QG.quantics_to_grididx(qgrid, init_pivots[1])
if order > 1
    ref = [quanticsf(QG.grididx_to_quantics(qgrid, (grididx_pivot1[1:end-1]..., p))) for p in 1:2^(R-10):2^R]
    reconst_tci = [ci(QG.grididx_to_quantics(qgrid, (grididx_pivot1[1:end-1]..., p))) for p in 1:2^(R-10):2^R]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(reconst_tci, label="QTCI", marker="x", linestyle="")
    ax.plot(ref, label="ref", marker="", linestyle="--")
    ax.set_xlabel(L"$τ_1$-Index $m$")
    ax.set_ylabel(L"f_m")
    ax.legend(frameon=false)
    plt.tight_layout()
    # fig.savefig("plots/comparsion_tau1scan_order$order.pdf")
    fig.savefig("comparsion_tau1scan_order$order.pdf")
end
ref = [quanticsf(QG.grididx_to_quantics(qgrid, (p, grididx_pivot1[2:end]...))) for p in 1:2^(R-10):2^R]
reconst_tci = [ci(QG.grididx_to_quantics(qgrid, (p, grididx_pivot1[2:end]...))) for p in 1:2^(R-10):2^R]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(reconst_tci, label="QTCI", marker="x", linestyle="")
ax.plot(ref, label="ref", marker="", linestyle="--")
ax.set_xlabel(L"$k_1$-Index $m$")
ax.set_ylabel(L"f_m")
ax.legend(frameon=false)
plt.tight_layout()
# fig.savefig("plots/comparsion_k1scan_order$order.pdf")
fig.savefig("comparsion_k1scan_order$order.pdf")


## Plots for bond dimensions and error analysis
fig, axs = plt.subplots(nrows=3, figsize=[10, 16])
axs[0].semilogy(1:length(ref), abs.(ref .- reconst_tci), label="|f_m - tci_m|")
axs[0].set_xlabel(L"$k_1$-Index $m$", fontsize=22)
axs[0].set_ylabel("interpolation error", fontsize=22)
axs[0].legend()

## Plot error vs bond dimension obtained by prrLU
# axs[1].plot(ranks, errors / ci.maxsamplevalue, marker="x")
axs[1].plot(ranks, errors, marker="x")
axs[1].set_xlabel("bond dimension", fontsize=22)
axs[1].set_ylabel("normalization error", fontsize=22)
axs[1].set_yscale("log")

function maxlinkdim(n::Integer, localdim::Integer=2)
    return 0:n-2, [min(localdim^i, localdim^(n - i)) for i in 1:(n-1)]
end
nquantics = var_dim * R

# axs[2].semilogy(1:nquantics-1, maxlinkdim(nquantics, 2)[2], color="gray", linewidth=0.5)
axs[2].plot(1:nquantics-1, [min(i * log(2), (nquantics - i) * log(2)) for i in 1:(nquantics-1)], color="gray", linewidth=0.5)
# axs[2].semilogy(1:nquantics-1, TCI.linkdims(ci))
D_max = maximum(bond_dims)
axs[2].plot(1:nquantics-1, log.(bond_dims))
axs[2].plot(1:nquantics-1, ones(nquantics - 1) * log(D_max), label=L"$D_{max}=%$(D_max)$", linestyle="--")
axs[2].set_ylabel(L"$\ln D_l$", fontsize=22)
axs[2].set_xlabel(L"$l$", fontsize=22)
# axs[2].set_xticks(1:2:nquantics-1)
axs[2].legend()
plt.tight_layout()

# fig.savefig("plots/bond_infos_order$order.pdf")
fig.savefig("bond_infos_order$order.pdf")
