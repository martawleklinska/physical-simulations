using LinearAlgebra
using CairoMakie

include("utils.jl")
include("ex1.jl")

## ex2: RTD
energy = collect(LinRange(0., 1., 500))
transmission = zeros(length(energy))
reflectance = zeros(length(energy))
for i in eachindex(energy)
    matrix = transfer_matrix_whole(energy[i] * EV_TO_HARTREE, x2, potential_double, mass_double)
    transmission[i] = get_transmission(energy[i] * EV_TO_HARTREE, generate_mass(x2, mass_double), matrix, x2, potential_double)
    reflectance[i] = get_reflectance(matrix)
end

fig = Figure();
ax2 = Axis(fig[1, 1], xlabel = "Energy (eV)", ylabel = "Transmission / Reflectance")

lines!(ax2, energy, transmission, color = :blue, label = "Transmission")
lines!(ax2, energy, reflectance, color = :red, label = "Reflectance")
Legend(fig, ax2, "Legend", position = :rb)
# display(fig)
save("ex2_trans_refl.pdf", fig)

## current--voltage

function current_voltage(V_bias::Float64)
    T::Float64 = 10
    mu_s::Float64 = 0.087
    μs = mu_s * EV_TO_HARTREE
    
    energy = collect(LinRange(1e-10, μs, 100))
    integral = 0.0
    ΔE = energy[2] - energy[1]
    for i in eachindex(energy)
        Ez = energy[i]
        V_fun = y -> potential_double_with_bias(y, V_bias * EV_TO_HARTREE)
        matrix = transfer_matrix_whole(Ez, x2, V_fun, mass_double)
        T_E = get_transmission(Ez, generate_mass(x2, mass_double), matrix, x2, V_fun)
        numerator = 1. + exp((μs - Ez) / (KBT * T))
        denominator = 1. + exp((μs - EV_TO_HARTREE * V_bias - Ez) / (KBT * T))
        integrand = T_E * log(numerator / denominator)
        integral += integrand * ΔE
    end
    prefactor = MASS_GAAS * KBT * T / (2 * π^2)
    return prefactor * integral
end

V_bias = collect(LinRange(0., 0.5, 50))
# negatywny opor rozniczkowy - tunelowanie rezonansowe

fig = Figure();
ax3 = Axis(fig[1, 1], xlabel = "V_bias (V)", ylabel = "I (A)")
current = zeros(length(V_bias))

for i in eachindex(V_bias)
    current[i] = current_voltage(V_bias[i])
end

lines!(ax3, V_bias, current, color = :blue, label = "Current")
Legend(fig, ax3, "Legend", position = :rb)
# display(fig)
save("ex2_current.pdf", fig)

println("Current: ", current)
