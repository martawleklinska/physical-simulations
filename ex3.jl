using LinearAlgebra
using CairoMakie

include("utils.jl")

function make_potential_QPC(left::Float64, right::Float64;
    dist::Float64 = 20.0 * NM_TO_BOHR,
    V_g1::Float64 = 4.0 * EV_TO_HARTREE,
    V_g2::Float64 = 4.0 * EV_TO_HARTREE)

    ε = 13.6
    constant = 0.5 / π / ε

    d = 3.0 * NM_TO_BOHR

    f(u, v) = constant * atan(u * v / (d * sqrt(d^2 + u^2 + v^2)))

    function g(x_pos, y_pos, l, b, t, r)
        f1 = f(x_pos - l, y_pos - b)
        f2 = f(x_pos - l, t - y_pos)
        f3 = f(r - x_pos, y_pos - b)
        f4 = f(r - x_pos, t - y_pos)
        return f1 + f2 + f3 + f4
    end

    l = left
    r = right

    b1 = dist / 2.0
    t1 = 10.0 * dist

    b2 = -dist / 2.0
    t2 = -10.0 * dist

    return (x, y) -> V_g1 * g(x, y, l, b1, t1, r) - V_g2 * g(x, y, l, b2, t2, r)
end

L = 100.0 * NM_TO_BOHR
W = 50.0  # in nm

left = 0.3 * L
right = 0.7 * L
dist = 0.6 * W * NM_TO_BOHR
x_qpc = collect(LinRange(0.0, L, 1000))


function compute_effective_potentials(x::Vector{Float64}, potential2D, nmax::Int;
    W_nm = 50.0, m_eff = 0.063)

    W = W_nm * NM_TO_BOHR
    y = collect(LinRange(-W/2, W/2, 200)) 
    dy = y[2] - y[1]
    N = length(y)

    coeff = (- 1.0)^2 / (2 * m_eff * dy^2)

    T = zeros(N, N)
    T[1, 1] = 2
    T[1, 2] = -1
    T[N, N] = 2
    T[N, N-1] = -1
    for i in 2:N-1
        T[i, i]     = 2
        T[i, i-1]   = -1
        T[i, i+1]   = -1
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

    return En_all  
end


qpc_potential_2D = make_potential_QPC(left, right; dist=dist)
En_all = compute_effective_potentials(x_qpc, qpc_potential_2D, 5)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x (nm)", ylabel = "E (eV)", title = "Efektywny potencjał Eₙ(x)")

for (n, En) in enumerate(En_all)
    lines!(ax, x_qpc / NM_TO_BOHR, En / EV_TO_HARTREE, label = "n = $n")
end
Legend(fig[1, 2], ax, "Legend")
display(fig)
save("ex3_potential.pdf", fig)


## conductance
include("ex1.jl")

function conductance(energy::Float64, number_of_states::Int64)
    potential = compute_effective_potentials(x_qpc, qpc_potential_2D, number_of_states)
    trans = 0
    for i in eachindex(potential)
        matrix = transfer_matrix_whole(energy * EV_TO_HARTREE, x_qpc, potential[i], mass_constant)
        transmission = get_transmission(energy * EV_TO_HARTREE, generate_mass(x_qpc, mass_constant), matrix, x_qpc, potential[i])
        trans += transmission
    end
    return trans
end
energy2 = collect(LinRange(0.0001, .065, 10))

transmis = zeros(length(energy2))
for i in eachindex(energy2)
    transmis[i] = conductance(energy2[i], 2)
    println("Energy: ", energy2[i], " eV, Conductance: ", transmis[i])
end
energy3 = collect(LinRange(0.065, .2, 40))

transmis2 = zeros(length(energy3))
for i in eachindex(energy3)
    transmis2[i] = conductance(energy3[i], 5)
    println("Energy: ", energy3[i], " eV, Conductance: ", transmis2[i])
end


fig = Figure();
ax2 = Axis(fig[1, 1], xlabel = "E (eV)", ylabel = "G (2e²/h)")

lines!(ax2, energy2, transmis, color = :blue, label = "Conductance")
lines!(ax2, energy3, transmis2, color = :blue, label = "Conductance")
Legend(fig[1, 2], ax2, "Legend")
display(fig)



## ex3
Vg_values = collect(LinRange(0.0, 20, 70)) 
conductance_vs_Vg = Dict{Float64, Vector{Float64}}()
fermi_energies = [50.0, 100.0] 

for E in fermi_energies
    conductances = Float64[]
    for Vg in Vg_values
        qpc_potential_2D = make_potential_QPC(left, right; dist=dist, V_g1=Vg * EV_TO_HARTREE, V_g2=Vg * EV_TO_HARTREE)
        
        En_all = compute_effective_potentials(x_qpc, qpc_potential_2D, 5)
        
        G = conductance(E, 5)  
        push!(conductances, G)
    end
    conductance_vs_Vg[E] = conductances
end

fig = Figure();
ax = Axis(fig[1, 1], xlabel = "Gate Voltage Vg (eV)", ylabel = "Conductance G (2e²/h)", title = "Conductance vs. Gate Voltage")

for E in fermi_energies
    lines!(ax, Vg_values, conductance_vs_Vg[E], label = "E = $(E) meV")
end

axislegend(ax)
display(fig)


