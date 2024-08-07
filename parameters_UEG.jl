## parameters of the uniform electron gas (UEG) model
order = 2 # Feynman diagrammatic expansion order (=loop number of Feynman diagrams)
# beta = 1.0
beta = 32.0 # Tf/T (Tf is the Fermi temperature)
rs = 2.0 # Wigner-Seitz radius
mass2 = 0.5 # Yukawa mass squared
Kdim = 3 # dimension of the k-space
var_dim = 4 * order - 1 # dimension of the variable space (k, θ, ϕ, τ)


println("Calculate the $order-order Σ(k=kF, ω=0)...")
println("Model: the $Kdim-dimensional Yukawa-interaction Electron Gas model (rs=$rs, Tf/T=$beta, λ=$mass2)")
