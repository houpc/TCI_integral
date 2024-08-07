using PythonCall: PythonCall
using PythonPlot: pyplot as plt, Figure
using PythonPlot
import QuanticsGrids as QG
using QuanticsTCI: quanticscrossinterpolate, integral
import TensorCrossInterpolation as TCI
using LaTeXStrings

PythonPlot.matplotlib.rcParams["font.size"] = 15

func(x) = 2^5 / (1 + 2 * sum(x))
var_dim = 5
integral_result = (-65205 * log(3) - 6250 * log(5) + 24010 * log(7) + 14641 * log(11)) / 24

R = 30 # number of bits
# R = 20 # number of bits
xmin = 0.0
xmax = 1.0
N = 2^R # size of the grid
tol = 1e-10 # Tolerance for the error

qgrid = QG.DiscretizedGrid{var_dim}(R, Tuple(fill(xmin, var_dim)), Tuple(fill(xmax, var_dim)); unfoldingscheme=:interleaved, includeendpoint=true)
quanticsf(sigma) = func(QG.quantics_to_origcoord(qgrid, sigma))

# Perform QTCI
# ci, ranks, errors = TCI.crossinterpolate2(Float64, quanticsf, QG.localdimensions(qgrid), init_pivots; tolerance=tol,
ci, ranks, errors = TCI.crossinterpolate2(Float64, quanticsf, QG.localdimensions(qgrid); tolerance=tol,
    normalizeerror=true, # Normalize the error by the maximum sample value,
    verbosity=1, loginterval=1, # Log the error every `loginterval` iterations
)
bond_dims = TCI.linkdims(ci)
println("Bond dimensions: $bond_dims")

integralvalue = TCI.sum(ci) * prod(QG.grid_step(qgrid))
println("QTCI result: $integralvalue")
println("Analytical result: $integral_result")


# Plot the function and the TCI reconstructions for k_1 scan
ref = [quanticsf(QG.grididx_to_quantics(qgrid, Tuple(rand(1:N, var_dim)))) for _ in 1:100]
reconst_tci = [ci(QG.grididx_to_quantics(qgrid, Tuple(rand(1:N, var_dim)))) for _ in 1:100]

fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot(ref, label="ref", marker="", linestyle="--")
ax.plot(reconst_tci, label="QTCI", marker="x", linestyle="")
# ax.plot(reconst_tci_globalpivot, label="TCI with global pivot", marker="+", linestyle="")
ax.set_xlabel(L"Index $m$")
ax.set_ylabel(L"f_m")
ax.legend(frameon=false)
plt.tight_layout()
fig.savefig("comparsion_ref_QTCI_test.pdf")


## Plots for bond dimensions and error analysis
fig, axs = plt.subplots(nrows=3, figsize=[10, 16])
axs[0].semilogy(1:length(ref), abs.(ref .- reconst_tci), label="log(|f_m - tci_m|)")
axs[0].set_xlabel(L"random index $m$", fontsize=22)
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

axs[2].plot(1:nquantics-1, [min(i * log(2), (nquantics - i) * log(2)) for i in 1:(nquantics-1)], color="gray", linewidth=0.5)
D_max = maximum(bond_dims)
axs[2].plot(1:nquantics-1, log.(bond_dims))
axs[2].plot(1:nquantics-1, ones(nquantics - 1) * log(D_max), label=L"$D_{max}=%$(D_max)$", linestyle="--")
axs[2].set_ylabel(L"$\ln D_l$", fontsize=22)
axs[2].set_xlabel(L"$l$", fontsize=22)
# axs[2].set_xticks(1:2:nquantics-1)
axs[2].legend()
plt.tight_layout()

fig.savefig("bond_infos_test.pdf")
