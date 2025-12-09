include("ex1.jl")

function potential_QPC(x::Float64, y::Float64, sigma_x::Float64, sigma_y::Float64,
    y_min::Float64, y_max::Float64, V_gates::Float64)
    exponential_term = -(x^2 / (sigma_x^2 + (y - y_min)^2 / (sigma_y^2)))^2
    exponential_term2 = -(x^2 / (sigma_x^2 + (y - y_max)^2 / (sigma_y^2)))^2

    V = -0.035 * V_gates * (exp(exponential_term) + exp(exponential_term2))

    return V    
end ## eV

function get_tau(N::Int64)
    matrix = zeros(N, N)
    for i in 1:N
        matrix[i, i] = -α
    end
    return matrix
end


function generalized_eigen_problem(N::Int64, E::Float64)
    matrix = zeros(ComplexF64, 2 * N, 2 * N)
    tau = get_tau(N)
    diagonal_term = 4 * α#  - α * (exp(im * kx * dx) + exp(-im * kx * dx))

    for i in 1:N
        matrix[N + i, i] = -tau[i, i]

        matrix[i, N + i] = 1

        matrix[N + i, N + i] = -diagonal_term + E
    end
    matrix[N+1, N+2] = α
    matrix[2*N, 2* N-1] = α
    for i in 2:N-1
        if i > 1
            matrix[N + i, N + i - 1] = α
        end
        if i < N
            matrix[N + i, N + i + 1] = α
        end
    end

    matrix_lambda = zeros(2 * N, 2 * N)
    for i in 1:N
            matrix_lambda[i, i] = 1
    end
    for i in 1:N
        matrix_lambda[N+i, N+i] = tau[i, i]
    end

    return matrix, matrix_lambda

end


### test if ok implemented
# A, B = generalized_eigen_problem(3, 0.2)

# eigen(A, B)
###

## E = 0.4
N = 19
A, B = generalized_eigen_problem(N, 0.4 * Hartree)
vals, vecs = eigen(A, B)
vecs = vecs[begin:N, :]

tol = 0.0000000000000005
indices = findall(val -> abs(abs(val) - 1) < tol, vals)

filtered_vals = vals[indices]                
filtered_vecs = vecs[:, indices]           

println("Filtered eigenvalues: ", filtered_vals)
println("Filtered eigenvectors: ", filtered_vecs)
function flip_every_second(v::Vector) ## we flip because u is 2N
    v_flipped = copy(v)
    for i in eachindex(v)
        if iseven(i)
            v_flipped[i] *= -1
        end
    end
    return v_flipped
end

modified_vecs = [flip_every_second(filtered_vecs[:, j]) for j in 1:size(filtered_vecs, 2)]
modified_vecs = hcat(modified_vecs...) 

function normalize_columns(M::AbstractMatrix)
    return [M[:, j] / norm(M[:, j]) for j in 1:size(M, 2)] |> x -> hcat(x...)
end
normalized_vecs = normalize_columns(filtered_vecs)


##
y = LinRange(y_min, y_max, N)
fig = Figure();
ax = Axis(fig[1,1], xlabel = "y [nm]", ylabel =  "u_n (a.u.)", title = "E = 0.4 eV")
lines!(ax, y, imag(filtered_vecs[:, 2]), label = "Im(u_2)")
lines!(ax, y, real(filtered_vecs[:, 2]), label = "Re(u_2)")
lines!(ax, y, imag(filtered_vecs[:, 1]), label = "Im(u_1)")
lines!(ax, y, real(filtered_vecs[:, 1]), label = "Re(u_1)")
Legend(fig[1, 2], ax, "")
# display(fig)
save("ex2_imag_real_energy04.pdf", fig)

##

## E = 0.2
N = 19
A2, B2 = generalized_eigen_problem(N, 0.2 * Hartree)
vals2, vecs2 = eigen(A2, B2)
vecs2 = vecs2[begin:N, :]

tol2 = 0.000000000000001
indices2 = findall(val2 -> abs(abs(val2) - 1) < tol, vals2)

filtered_vals2 = vals2[indices2]                
filtered_vecs2 = vecs2[:, indices2]           

function flip_every_second(v::Vector)
    v_flipped = copy(v)
    for i in eachindex(v)
        if iseven(i)
            v_flipped[i] *= -1
        end
    end
    return v_flipped
end

modified_vecs2 = [flip_every_second(filtered_vecs2[:, j]) for j in 1:size(filtered_vecs2, 2)]
modified_vecs2 = hcat(modified_vecs2...) 
normalized_vecs2 = normalize_columns(modified_vecs2)
##
y = LinRange(y_min, y_max, N)
fig = Figure();
ax = Axis(fig[1,1], xlabel = "y [nm]", ylabel =  "u_n (a.u.)", title = "E = 0.2 eV")
lines!(ax, y, -real(filtered_vecs2[:, 1]), label = "Re(u_1)")
lines!(ax, y, imag(filtered_vecs2[:, 1]), label = "Im(u_1)")
Legend(fig[1, 2], ax, "")
# display(fig)
save("ex2_imag_real_values_energy02.pdf", fig)

## calculating v


function get_velocity(eigen_value, eigen_vector)
    tau = get_tau(N)
    constants = - 2 * dx 
    imgainary = imag(eigen_value * eigen_vector' * tau' * eigen_vector)
    return imgainary * constants
end

a = get_velocity(filtered_vals2[2], filtered_vecs2[:, 2])

## getting kx

function get_kx(eigen_value)
    result = log(eigen_value) / (im * dx)
    return result
end

a = get_kx(filtered_vals2[1])
a2 = get_kx(filtered_vals[1])
a3 = get_kx(filtered_vals[2])

## plotting dispersion relation with kx
function exercise2_energy02()
    kx_vals = LinRange(-π / dx, π / dx, 100)
    energies = zeros(length(kx_vals), Ny)

    for (i, kx) in enumerate(kx_vals)
        energies[i, :] = dispersion_jednorodny_kanal(mass_InSb, kx) 
    end

    fig = Figure();
    ax = Axis(fig[1, 1], xlabel = "kx (nm)", ylabel = "E (eV)", title = " Eₙ(x), E = 0.2 eV")
    xlims!(ax, -2, 2)
    ylims!(ax, 0., 5)
    cm = cgrad(:managua10, 19)
    
    lines!(ax, kx_vals*nm_to_bohr, energies[:, 1]/Hartree, color = cm[1])
    for n in 2:19
        lines!(ax, kx_vals*nm_to_bohr, energies[:, n]/Hartree, label="n=$n", color = cm[n])
    end
    scatter!(ax, -real(a) * nm_to_bohr, 0.2)
    scatter!(ax, real(a) * nm_to_bohr, 0.2)
    # display(fig)
    save("ex2_disperssion_energy02.pdf", fig)
end
exercise2_energy02()


function exercise2_energy04()
    kx_vals = LinRange(-π / dx, π / dx, 100)
    energies = zeros(length(kx_vals), Ny)

    for (i, kx) in enumerate(kx_vals)
        energies[i, :] = dispersion_jednorodny_kanal(mass_InSb, kx) 
    end

    fig = Figure();
    ax = Axis(fig[1, 1], xlabel = "kx (nm)", ylabel = "E (eV)", title = " Eₙ(x), E = 0.4 eV")
    xlims!(ax, -2, 2)
    ylims!(ax, 0., 5)
    cm = cgrad(:managua10, 19)
    
    lines!(ax, kx_vals*nm_to_bohr, energies[:, 1]/Hartree, color = cm[1])
    for n in 2:19
        lines!(ax, kx_vals*nm_to_bohr, energies[:, n]/Hartree, label="n=$n", color = cm[n])
    end
    scatter!(ax, real(a2) * nm_to_bohr, 0.4)
    scatter!(ax, -real(a2) * nm_to_bohr, 0.4)
    scatter!(ax, -real(a3) * nm_to_bohr, 0.4)
    scatter!(ax, real(a3) * nm_to_bohr, 0.4)
    # display(fig)
    save("ex2_disperssion_energy04.pdf", fig)
end
exercise2_energy04()