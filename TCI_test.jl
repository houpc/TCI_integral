using PythonCall: PythonCall
using PythonPlot: pyplot as plt, Figure

# Displays the matplotlib figure object `fig` and avoids duplicate plots.
_display(fig::Figure) = isinteractive() ? (fig; plt.show(); nothing) : Base.display(fig)
_display(fig::PythonCall.Py) = _display(Figure(fig))

import QuanticsGrids as QG
using QuanticsTCI: quanticscrossinterpolate, integral
import TensorCrossInterpolation as TCI

# B = 2^(-30) # global variable
# function f(x)
#     return cos(x / B) * cos(x / (4 * sqrt(5) * B)) * exp(-x^2) + 2 * exp(-x)
# end
# Define callable struct
Base.@kwdef struct Ritter2024 <: Function
    B::Float64 = 2^(-30)
end

# Make Ritter2024 be "callable" object.
function (obj::Ritter2024)(x)
    B = obj.B
    return cos(x / B) * cos(x / (4 * sqrt(5) * B)) * exp(-x^2) + 2 * exp(-x)
end

f = Ritter2024()
nothing # hide


xs = LinRange(0, 2.0^(-23), 1000)

fig, ax = plt.subplots()
ax.plot(xs, f.(xs), label="$(nameof(f))")
ax.set_title("$(nameof(f))")
ax.legend()
_display(fig)

R = 30 # number of bits
xmin = 0.0
xmax = log(20.0)
N = 2^R # size of the grid
# * Uniform grid (includeendpoint=false, default):
#   -xmin, -xmin+dx, ...., -xmin + (2^R-1)*dx
#     where dx = (xmax - xmin)/2^R.
#   Note that the grid does not include the end point xmin.
#
# * Uniform grid (includeendpoint=true):
#   -xmin, -xmin+dx, ...., xmin-dx, xmin,
#     where dx = (xmax - xmin)/(2^R-1).
qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
# Convert to quantics format and sweep
tol = 1e-8 # Tolerance for the error
ci, ranks, errors = quanticscrossinterpolate(
    Float64, f, qgrid;
    tolerance=tol,
    maxbonddim=15,
    normalizeerror=true, # Normalize the error by the maximum sample value,
    verbosity=1, loginterval=1, # Log the error every `loginterval` iterations
)
println(errors ./ ci.tci.maxsamplevalue)
println(TCI.linkdims(ci.tci))

for i in [1, 2, 3, 2^R] # Linear indices
    # restore original coordinate `x` from linear index `i`
    x = QG.grididx_to_origcoord(qgrid, i)
    println("x: $(x), i: $(i), tci: $(ci(i)), ref: $(f(x))")
end

maxindex = QG.origcoord_to_grididx(qgrid, 2.0^(-23))
testindices = Int.(round.(LinRange(1, maxindex, 1000)))

xs = [QG.grididx_to_origcoord(qgrid, i) for i in testindices]
ys = f.(xs)
yci = ci.(testindices)

fig, ax = plt.subplots()
ax.plot(xs, ys, label="$(nameof(f))")
ax.plot(xs, yci, label="tci", linestyle="dashed", alpha=0.7)
ax.set_title("$(nameof(f)) and TCI")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
_display(fig)

fig, ax = plt.subplots()

ys = f.(xs)
yci = ci.(testindices)

ax.semilogy(xs, abs.(ys .- yci), label="log(|f(x) - ci(x)|)")

ax.set_title("x vs interpolation error: $(nameof(f))")
ax.set_xlabel("x")
ax.set_ylabel("interpolation error")
ax.legend()
_display(fig)

println(integral(ci), 19 / 10)
print(sum(ci) * (log(20) - 0) / 2^R, 19 / 10)

# Plot error vs bond dimension obtained by prrLU
fig, ax = plt.subplots()
ax.plot(ci.tci.pivoterrors ./ ci.tci.maxsamplevalue, marker="x")
ax.set_xlabel("Bond dimension")
ax.set_ylabel("Normalization error")
ax.set_title("normalized error vs. bond dimension: $(nameof(f))")
ax.set_yscale("log")
_display(fig)


R = 20 # number of bits
N = 2^R  # size of the grid

qgrid = QG.DiscretizedGrid{1}(R, -10, 10; includeendpoint=false)

# Function of interest
function oscillation_fn(x)
    return (
        sinc(x) + 3 * exp(-0.3 * (x - 4)^2) * sinc(x - 4) - cos(4 * x)^2 -
        2 * sinc(x + 10) * exp(-0.6 * (x + 9)) + 4 * cos(2 * x) * exp(-abs(x + 5)) +
        6 * 1 / (x - 11) + sqrt(abs(x)) * atan(x / 15))
end

# Convert to quantics format and sweep
ci, ranks, errors = quanticscrossinterpolate(Float64, oscillation_fn, qgrid; maxbonddim=20, verbosity=1, loginterval=1)
tol = 1e-10 # Tolerance for the error

# Convert to quantics format and sweep
ci_tol, ranks_tol, errors_tol = quanticscrossinterpolate(
    Float64, oscillation_fn, qgrid;
    tolerance=tol,
    normalizeerror=true, # Normalize the error by the maximum sample value,
    verbosity=1, loginterval=1, # Log the error every `loginterval` iterations
)
println(errors_tol ./ ci_tol.tci.maxsamplevalue)