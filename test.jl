function compute_effective_potentials(x::Vector{Float64}, potential2D, nmax::Int;
    W_nm = 50.0, m_eff = 0.063)

    W = W_nm * NM_TO_BOHR
    y = collect(LinRange(-W/2, W/2, 200))  # transverse grid
    dy = y[2] - y[1]
    N = length(y)

    coeff = - 1.0^2 / (2 * m_eff * dy^2)

    # kinetic energy matrix (finite difference)
    T = zeros(N, N)
    for i in 2:N-1
        T[i, i]     = -2
        T[i, i-1]   = 1
        T[i, i+1]   = 1
    end
    T *= coeff

    En_all = [zeros(length(x)) for _ in 1:nmax]

    for (i, xval) in enumerate(x)
        V_y = [potential2D(xval, yval) for yval in y]
        V = Diagonal(V_y)
        H = T + V
        evals = eigen(H).values
        for n in 1:nmax
            En_all[n][i] = evals[n]
        end
    end

    return En_all  # list of vectors E₁(x), E₂(x), ...
end

qpc_potential_2D = make_potential_QPC(left, right; dist=dist)
En_all = compute_effective_potentials(x_qpc, qpc_potential_2D, 5)

# Then plot as before:
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x (nm)", ylabel = "Energy (eV)", title = "Effective Potentials Eₙ(x)")

for (n, En) in enumerate(En_all)
    lines!(ax, x_qpc / NM_TO_BOHR, En * EV_TO_HARTREE, label = "n = $n")
end
display(fig)
