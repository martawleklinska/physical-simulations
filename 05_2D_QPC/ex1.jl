
using LinearAlgebra
using CairoMakie

mass_InSb = 0.017
sigma_x = 300 # nm
sigma_y = 300 # nm

Hartree = 1/27.2114 
nm_to_bohr = 1.0 / 0.0529177

function potential_zero(x::Float64, y::Float64)
    return 0.0
end


mass_InSb = 0.017 
Ny = 19
y_min = -10.0               # nm
y_max = 10.0                # nm
dx = (y_max - y_min) / (Ny + 1)  # nm
dx *= nm_to_bohr
α = 1 / (2 * mass_InSb * dx^2)

function dispersion_jednorodny_kanal(mass::Float64, kx::Float64)
    α = 1 / (2 * mass * dx^2)
    diagonal_term = 4 * α - α * (exp(im * kx * dx) + exp(-im * kx * dx))

    H = zeros(ComplexF64, Ny, Ny)
    for i in 1:Ny
        H[i, i] = diagonal_term
        if i > 1
            H[i, i - 1] = -α
        end
        if i < Ny
            H[i, i + 1] = -α
        end
    end

    evals = eigvals(H)
    return sort(real(evals))  
end

function exercise1()
    kx_vals = LinRange(-π / dx, π / dx, 100)
    energies = zeros(length(kx_vals), Ny)

    for (i, kx) in enumerate(kx_vals)
        energies[i, :] = dispersion_jednorodny_kanal(mass_InSb, kx) 
    end

    fig = Figure();
    ax = Axis(fig[1, 1], xlabel = "kx (nm)", ylabel = "E (eV)", title = " Eₙ(x)")
    cm = cgrad(:managua10, 19)
    
    lines!(ax, kx_vals*nm_to_bohr, energies[:, 1]/Hartree, color = cm[1])
    for n in 2:19
        lines!(ax, kx_vals*nm_to_bohr, energies[:, n]/Hartree, label="n=$n", color = cm[n])
    end
    # display(fig)
    save("ex1_disperssion_relation.pdf", fig)
end

exercise1()
