using LinearAlgebra
using CairoMakie

include("utils.jl")

function transfer_matrix(k::Vector{Complex{Float64}}, z::Vector{Float64}, n::Int64, mass::Vector{Float64})
    T::Matrix{Complex{Float64}} = zeros(2, 2)
    T[1, 1] = 0.5 * (1 + (k[n + 1] * mass[n]) / (k[n] * mass[n + 1])) * exp(im * (k[n + 1] - k[n]) * z[n])
    T[1, 2] = 0.5 * (1 - (k[n + 1] * mass[n]) / (k[n] * mass[n + 1])) * exp(-im * (k[n + 1] + k[n]) * z[n]) 
    T[2, 1] = 0.5 * (1 - (k[n + 1] * mass[n]) / (k[n] * mass[n + 1])) * exp(im * (k[n + 1] + k[n]) * z[n]) 
    T[2, 2] = 0.5 * (1 + (k[n + 1] * mass[n]) / (k[n] * mass[n + 1])) * exp(-im * (k[n + 1] - k[n]) * z[n])
    return T
end


function transfer_matrix_whole(energy::Float64, x::Vector{Float64}, potential::Function, mass::Function)
    N = length(x) - 1
    k::Vector{Complex{Float64}} = wave_number(energy, generate_potential(x,  potential), generate_mass(x, mass))
    matrix::Matrix{Complex{Float64}} = transfer_matrix(k, x, 1, generate_mass(x, mass))
    for n in 2:N
        matrix *= transfer_matrix(k, x, n, generate_mass(x, mass))  
    end
    return matrix
end

function transfer_matrix_whole(energy::Float64, x::Vector{Float64}, potential::Vector{Float64}, mass::Function)
    N = length(x) - 1
    k::Vector{Complex{Float64}} = wave_number(energy, potential, generate_mass(x, mass))
    matrix::Matrix{Complex{Float64}} = transfer_matrix(k, x, 1, generate_mass(x, mass))
    for n in 2:N
        matrix *= transfer_matrix(k, x, n, generate_mass(x, mass))  
    end
    return matrix
end

function get_transmission(energy::Float64, mass::Vector{Float64}, matrix::Matrix{Complex{Float64}}, x::Vector{Float64}, potential::Function)
    k = wave_number(energy, [potential(x[1]), potential(x[end])], [mass[1], mass[end]])
    k1 = abs(k[1])
    kN = abs(k[end])
    mass1 = mass[1]
    massN = mass[end]
    matrix_el_11 = matrix[1, 1]
    transission = real(kN) * mass1 / (real(k1) * massN) / abs(matrix_el_11)^2
    return transission
end

function get_transmission(energy::Float64, mass::Vector{Float64}, matrix::Matrix{Complex{Float64}}, x::Vector{Float64}, potential::Vector{Float64})
    k = wave_number(energy, [potential[1], potential[end]], [mass[1], mass[end]])
    k1 = abs(k[1])
    kN = abs(k[end])
    mass1 = mass[1]
    massN = mass[end]
    matrix_el_11 = matrix[1, 1]
    transission = real(kN) * mass1 / (real(k1) * massN) / abs(matrix_el_11)^2
    return transission
end

function get_reflectance(matrix::Matrix{Complex{Float64}})
    matrix_el_11 = matrix[1, 1]
    matrix_el_21 = matrix[2, 1]
    reflectance = abs(matrix_el_21)^2 / abs(matrix_el_11)^2
    return reflectance
end

## ex1: Transmission and Reflectance
energy = collect(LinRange(0., 1., 100))
transmission = zeros(length(energy))
reflectance = zeros(length(energy))
for i in eachindex(energy)
    matrix = transfer_matrix_whole(energy[i] * EV_TO_HARTREE, x, potential_single, mass_single)
    transmission[i] = get_transmission(energy[i] * EV_TO_HARTREE, generate_mass(x, mass_single), matrix, x, potential_single)
    reflectance[i] = get_reflectance(matrix)
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Energy (eV)", ylabel = "Transmission / Reflectance")

lines!(ax, energy, transmission, color = :blue, label = "Transmission")
lines!(ax, energy, reflectance, color = :red, label = "Reflectance")
Legend(fig, ax, "Legend", position = :right)
# display(fig)
save("ex1_trans_refl_const.pdf", fig)
