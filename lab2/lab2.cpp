#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

const double HARTREE_ENERGY = 27.211;
const double LENGTH_SCALE = 10.0 / 0.052918;
const double EFFECTIVE_MASS = 0.067;
const double DIELECTRIC_CONSTANT = 12.5;
const double CONVERGENCE_TOL = 1e-6;
const int GRID_SIZE = 41;

// Electron interaction function
double electron_interaction(const VectorXd &x_values, int i, int j, double dielectric_const, double wire_thickness) {
    return 1.0 / (dielectric_const * sqrt(pow(x_values[i] - x_values[j], 2) + wire_thickness * wire_thickness));
}

// Compute Hamiltonian function
MatrixXd compute_hamiltonian(const MatrixXd &wavefunction, const VectorXd &x_values, double mass, double dx, double dielectric_const, double wire_thickness) {
    MatrixXd hamiltonian = MatrixXd::Zero(GRID_SIZE, GRID_SIZE);
    
    for (int i = 1; i < GRID_SIZE - 1; ++i) {
        for (int j = 1; j < GRID_SIZE - 1; ++j) {
            hamiltonian(i, j) = -(wavefunction(i + 1, j) + wavefunction(i - 1, j) + wavefunction(i, j + 1) + wavefunction(i, j - 1) - 4 * wavefunction(i, j)) / (2.0 * mass * dx * dx);
            hamiltonian(i, j) += wavefunction(i, j) * electron_interaction(x_values, i, j, dielectric_const, wire_thickness);
        }
    }
    return hamiltonian;
}

// Normalize wavefunction
void normalize_wavefunction(MatrixXd &wavefunction, double dx) {
    double norm_factor = sqrt((wavefunction.array().square().sum()) * dx * dx);
    wavefunction /= norm_factor;
}

// Compute energy
double compute_energy(const MatrixXd &wavefunction, const MatrixXd &hamiltonian, double dx) {
    return (wavefunction.array() * hamiltonian.array()).sum() * dx * dx;
}

// Convergence test
bool convergence_test(double prev_energy, double current_energy, double tolerance) {
    return fabs((current_energy - prev_energy) / current_energy) < tolerance;
}

// Imaginary time evolution
MatrixXd imaginary_time_evolution(double box_size, double electron_mass, double dielectric_const, double wire_thickness, double tolerance) {
    double dx = 2 * box_size / (GRID_SIZE - 1);
    VectorXd x_values = VectorXd::LinSpaced(GRID_SIZE, -box_size, box_size);
    double time_step = electron_mass * dx * dx * 0.4;
    
    MatrixXd wavefunction = MatrixXd::Random(GRID_SIZE, GRID_SIZE);
    wavefunction.block(1, 1, GRID_SIZE - 2, GRID_SIZE - 2) = MatrixXd::Random(GRID_SIZE - 2, GRID_SIZE - 2);
    
    double energy = 0;
    bool converged = false;
    
    while (!converged) {
        MatrixXd hamiltonian = compute_hamiltonian(wavefunction, x_values, electron_mass, dx, dielectric_const, wire_thickness);
        MatrixXd updated_wavefunction = wavefunction - time_step * hamiltonian;
        normalize_wavefunction(updated_wavefunction, dx);
        
        double new_energy = compute_energy(updated_wavefunction, hamiltonian, dx);
        converged = convergence_test(energy, new_energy, tolerance);
        
        wavefunction = updated_wavefunction;
        energy = new_energy;
    }
    
    return wavefunction;
}

int main() {
    double box_size = 30 / 0.052918;
    MatrixXd wavefunction = imaginary_time_evolution(box_size, EFFECTIVE_MASS, DIELECTRIC_CONSTANT, LENGTH_SCALE, CONVERGENCE_TOL);
    
    cout << "Wavefunction computed successfully!" << endl;
    return 0;
}
