
MASS_GAAS = 0.063
# MASS_ALGAAS = (0.063 + 0.083 * 0.3)
MASS_ALGAAS = (0.063)# + 0.083 * 0.3) # with constant mass
HARTREE_ENERGY = 27.2114
NM_TO_BOHR = 1 / 0.05291772108
EV_TO_HARTREE = 1 / HARTREE_ENERGY
KBT = 3.167e-6

## ex1
d1 = 3. * NM_TO_BOHR
d2 = 8. * NM_TO_BOHR
d3 = 11. * NM_TO_BOHR
U_barrier = 0.27 * EV_TO_HARTREE

xmin = 0. * NM_TO_BOHR  
xmax = d3
x = collect(LinRange(xmin, xmax, 1000))

## ex2 -> first barrier from the 1st ex and d3:: beginning of next barrier; 
    # d4:: end of the next barrier; d5:: end of system
d4 = 16 * NM_TO_BOHR
d5 = 19 * NM_TO_BOHR

x2 = collect(LinRange(xmin, d5, 1000))

##
function potential_single(x::Float64)
    if x < d1 || x > d2
        return 0.0
    else 
        return U_barrier
    end
end

function potential_double(x::Float64)
    if (x < d1) || (x > d2 && x < d3) || (x > d4)
        return 0.0
    else 
        return U_barrier
    end
end

function potential_double_with_bias(x::Float64, bias::Float64)
    potential = 0.0
    if (x < d1) || (x > d2 && x < d3) || (x > d4)
        potential = -x * bias/d5
    else 
        potential = U_barrier - x * bias/d5
    end
    return potential
end

function mass_single(x::Float64)
    if (x < d1 || x > d2)
        return MASS_GAAS
    else
        return MASS_ALGAAS
    end
end

function mass_constant(x::Float64)
    return MASS_GAAS
end
function mass_double(x::Float64)
    if (x < d1) || (x > d2 && x < d3) || (x > d4)
        return MASS_GAAS
    else
        return MASS_ALGAAS
    end
end

function generate_potential(x::Vector{Float64}, potential::Function)
    V = zeros(length(x))
    for i in 1:length(x)
        V[i] = potential(x[i])
    end
    return V
end

function generate_mass(x::Vector{Float64}, mass)
    m = zeros(length(x))
    for i in 1:length(x)
        m[i] = mass(x[i])
    end
    return m
end

function wave_number(E, V, mass)
    k_vector = Complex{Float64}[]
    for i in eachindex(V)
        v = V[i]
        if (E > v)
            push!(k_vector, sqrt(2 * mass[i] * (E - v)))
        else
            push!(k_vector, im * sqrt(2 * mass[i] * (v - E)))
        end
    end
    return k_vector
end
