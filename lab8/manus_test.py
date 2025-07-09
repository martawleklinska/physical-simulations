import numpy as np
import kwant
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class KwantBaseTask:
    """Base class for KWANT tasks with common functionality."""

    def __init__(self, params=None):
        """Initialize with default or custom parameters."""
        # Default parameters
        self.default_params = {
            "m": 1,  # electron mass
            "mu": 10.0,  # chemical potential (meV)
            "delta": 0.25,  # superconducting gap (meV)
            "dx": 0.2,  # lattice spacing (nm)
            "a": 1.0,  # scattering potential width (nm)
            "L_normal": 250,  # length of normal/ferromagnetic region (nm)
            "L_sc": 250,  # length of superconducting region (nm)
            "P": 0.0,  # spin polarization (0 to 1)
            "Z": 0.0,  # interface barrier strength
            "energy_range": [0, 0.5],  # energy range for calculations (meV)
        }

        # Update with custom parameters if provided
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)

        # Create sigma matrices for conservation law
        self.sigma_law = np.array([[1, 0], [0, 2]])

        # Initialize system
        self.system = None

    def make_onsite(self, site, h_func, delta_func, V_func):
        """
        Create onsite Hamiltonian matrix (2x2) for a given site.

        Args:
            site: KWANT site object
            h_func: Function returning exchange field at position
            delta_func: Function returning superconducting gap at position
            V_func: Function returning scattering potential at position

        Returns:
            2x2 matrix for onsite energy
        """
        (x,) = site.pos
        m = self.params["m"]
        mu = self.params["mu"]
        dx = self.params["dx"]

        # Get position-dependent parameters
        h = h_func(x)
        delta = delta_func(x)
        V = V_func(x)

        # Kinetic term from discretization of -ħ²∇²/2m
        kinetic = 1.0 / (m * dx**2)

        # Construct onsite matrix according to Bogoliubov-de Gennes equation
        onsite = np.array(
            [[2 * kinetic + V - mu - h, delta], [delta, -2 * kinetic - V + mu + h]]
        )

        return onsite

    def make_hopping(self, site1, site2, h_func, delta_func):
        """
        Create hopping Hamiltonian matrix (2x2) between sites.

        Args:
            site1, site2: KWANT site objects
            h_func: Function returning exchange field at position
            delta_func: Function returning superconducting gap at position

        Returns:
            2x2 matrix for hopping energy
        """
        m = self.params["m"]
        dx = self.params["dx"]

        # Kinetic term from discretization
        t = -1.0 / (2 * m * dx**2)

        # Construct hopping matrix
        hopping = np.array([[t, 0], [0, -t]])

        return hopping

    def build_system(self):
        """Build the KWANT system (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement build_system method")

    def calculate_conductance(self, energy):
        """Calculate conductance at given energy."""
        raise NotImplementedError(
            "Subclasses must implement calculate_conductance method"
        )

    def run_simulation(self):
        """Run the simulation (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement run_simulation method")

    def save_plot(self, x_data, y_data, xlabel, ylabel, title, filename, legend=None):
        """Save plot to file."""
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        if isinstance(y_data[0], (list, np.ndarray)):
            for i, y in enumerate(y_data):
                ax.plot(x_data, y, label=legend[i] if legend else f"Series {i}")
            if legend:
                ax.legend()
        else:
            ax.plot(x_data, y_data)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

        fig.savefig(filename)
        return filename


class NormalSuperconductorTask(KwantBaseTask):
    """Task for simulating Normal Metal (Ferromagnet)/Superconductor junction."""

    def __init__(self, params=None):
        super().__init__(params)
        self.name = "NM_SC_Junction"

    def h_function(self, x):
        """Exchange field as a function of position."""
        P = self.params["P"]
        mu = self.params["mu"]
        L_normal = self.params["L_normal"]

        # Exchange field is P*mu in ferromagnet region, 0 in superconductor
        if x < L_normal:
            return P * mu
        else:
            return 0.0

    def delta_function(self, x):
        """Superconducting gap as a function of position."""
        delta = self.params["delta"]
        L_normal = self.params["L_normal"]

        # Delta is 0 in normal/ferromagnet region, delta in superconductor
        if x < L_normal:
            return 0.0
        else:
            return delta

    def V_function(self, x):
        """Scattering potential at the interface."""
        Z = self.params["Z"]
        mu = self.params["mu"]
        a = self.params["a"]
        L_normal = self.params["L_normal"]

        # Gaussian barrier at the interface
        return Z * mu * np.exp(-((x - L_normal) ** 2) / (2 * a**2))

    def build_system(self):
        """Build the NM/SC junction system."""
        dx = self.params["dx"]
        L_normal = self.params["L_normal"]
        L_sc = self.params["L_sc"]

        # Define 1D lattice with 2 orbitals per site
        lat = kwant.lattice.chain(dx, norbs=2)

        # Create system
        system = kwant.Builder()

        # Define onsite and hopping functions
        def onsite(site):
            return self.make_onsite(
                site, self.h_function, self.delta_function, self.V_function
            )

        def hopping(site1, site2):
            return self.make_hopping(site1, site2, self.h_function, self.delta_function)

        # Add sites and hoppings to the system
        # Normal/ferromagnet region
        normal_sites = [lat(i) for i in range(int(L_normal / dx))]
        for site in normal_sites:
            system[site] = onsite

        # Superconductor region
        sc_sites = [
            lat(i) for i in range(int(L_normal / dx), int((L_normal + L_sc) / dx))
        ]
        for site in sc_sites:
            system[site] = onsite

        # Add hoppings between all adjacent sites
        all_sites = normal_sites + sc_sites
        for i in range(len(all_sites) - 1):
            system[all_sites[i], all_sites[i + 1]] = hopping

        # Add left lead (normal/ferromagnet)
        sym_left = kwant.TranslationalSymmetry((-dx,))
        left_lead = kwant.Builder(sym_left, conservation_law=self.sigma_law)

        left_lead[lat(0)] = onsite
        left_lead[lat(0), lat(1)] = hopping

        # Add right lead (superconductor)
        sym_right = kwant.TranslationalSymmetry((dx,))
        right_lead = kwant.Builder(sym_right)

        right_lead[lat(0)] = onsite
        right_lead[lat(0), lat(1)] = hopping

        # Attach leads
        system.attach_lead(left_lead)
        system.attach_lead(right_lead)

        # Finalize the system
        self.system = system.finalized()
        return self.system

    def calculate_conductance(self, energy):
        """Calculate conductance at given energy."""
        if self.system is None:
            self.build_system()

        # Calculate scattering matrix
        smatrix = kwant.smatrix(self.system, energy)

        # Calculate reflection probabilities
        # (0,0) -> electron from left lead to electron in left lead (normal reflection)
        Ree = smatrix.transmission((0, 0), (0, 0))

        # (0,1) -> electron from left lead to hole in left lead (Andreev reflection)
        Rhe = smatrix.transmission((0, 1), (0, 0))

        # Calculate transmission probability
        # Electron from left lead to quasiparticle in right lead
        T = smatrix.transmission(1, 0)

        # Calculate conductance using formula G = (e²/h) * (1 - Ree + Rhe)
        G = 1.0 - Ree + Rhe

        return G, Ree, Rhe, T

    def run_simulation(self):
        """Run the simulation for NM/SC junction."""
        results = {}

        # Build the system
        self.build_system()

        # Task 1: Calculate conductance vs energy for different Z values
        energy_min, energy_max = self.params["energy_range"]
        energies = np.linspace(energy_min, energy_max, 100)

        # For different Z values
        Z_values = [0.0, 0.5, 1.0, 1.5]
        conductances = []
        Ree_values = []
        Rhe_values = []
        T_values = []

        for Z in Z_values:
            self.params["Z"] = Z
            self.build_system()

            G_vs_E = []
            Ree_vs_E = []
            Rhe_vs_E = []
            T_vs_E = []

            for E in energies:
                G, Ree, Rhe, T = self.calculate_conductance(E)
                G_vs_E.append(G)
                Ree_vs_E.append(Ree)
                Rhe_vs_E.append(Rhe)
                T_vs_E.append(T)

            conductances.append(G_vs_E)
            Ree_values.append(Ree_vs_E)
            Rhe_values.append(Rhe_vs_E)
            T_values.append(T_vs_E)

        results["Z_study"] = {
            "energies": energies,
            "Z_values": Z_values,
            "conductances": conductances,
            "Ree": Ree_values,
            "Rhe": Rhe_values,
            "T": T_values,
        }

        # Reset Z to 0 for polarization study
        self.params["Z"] = 0.0

        # Task 2: Calculate conductance vs energy for different P values
        P_values = [0.0, 0.5, 0.8, 0.99]
        conductances = []
        Ree_values = []
        Rhe_values = []
        T_values = []

        for P in P_values:
            self.params["P"] = P
            self.build_system()

            G_vs_E = []
            Ree_vs_E = []
            Rhe_vs_E = []
            T_vs_E = []

            for E in energies:
                G, Ree, Rhe, T = self.calculate_conductance(E)
                G_vs_E.append(G)
                Ree_vs_E.append(Ree)
                Rhe_vs_E.append(Rhe)
                T_vs_E.append(T)

            conductances.append(G_vs_E)
            Ree_values.append(Ree_vs_E)
            Rhe_values.append(Rhe_vs_E)
            T_values.append(T_vs_E)

        results["P_study"] = {
            "energies": energies,
            "P_values": P_values,
            "conductances": conductances,
            "Ree": Ree_values,
            "Rhe": Rhe_values,
            "T": T_values,
        }

        # Task 3: Calculate coefficients vs P for E near zero
        P_fine = np.linspace(0, 0.99999, 100)
        E_near_zero = 1e-6

        Ree_vs_P = []
        Rhe_vs_P = []
        T_vs_P = []

        for P in P_fine:
            self.params["P"] = P
            self.build_system()

            G, Ree, Rhe, T = self.calculate_conductance(E_near_zero)
            Ree_vs_P.append(Ree)
            Rhe_vs_P.append(Rhe)
            T_vs_P.append(T)

        results["P_fine_study"] = {
            "P_values": P_fine,
            "Ree": Ree_vs_P,
            "Rhe": Rhe_vs_P,
            "T": T_vs_P,
        }

        return results


class FerromagnetSuperconductorFerromagnetTask(KwantBaseTask):
    """Task for simulating Ferromagnet/Superconductor/Ferromagnet junction."""

    def __init__(self, params=None):
        super().__init__(params)
        self.name = "FM_SC_FM_Junction"

        # Additional parameters for FM/SC/FM junction
        additional_params = {
            "P_left": 0.0,  # left ferromagnet polarization
            "P_right": 0.0,  # right ferromagnet polarization
            "dx": 1.0,  # larger lattice spacing for this task
        }

        # Update parameters
        self.params.update(additional_params)
        if params:
            self.params.update(params)

    def h_function(self, x):
        """Exchange field as a function of position."""
        P_left = self.params["P_left"]
        P_right = self.params["P_right"]
        mu = self.params["mu"]
        L_normal = self.params["L_normal"]
        L_sc = self.params["L_sc"]

        # Exchange field in left FM, SC, and right FM regions
        if x < L_normal:
            return P_left * mu  # Left ferromagnet
        elif x < L_normal + L_sc:
            return 0.0  # Superconductor
        else:
            return P_right * mu  # Right ferromagnet

    def delta_function(self, x):
        """Superconducting gap as a function of position."""
        delta = self.params["delta"]
        L_normal = self.params["L_normal"]
        L_sc = self.params["L_sc"]

        # Delta is 0 in ferromagnet regions, delta in superconductor
        if L_normal <= x < L_normal + L_sc:
            return delta
        else:
            return 0.0

    def V_function(self, x):
        """Scattering potential at the interfaces."""
        Z = self.params["Z"]
        mu = self.params["mu"]
        a = self.params["a"]
        L_normal = self.params["L_normal"]
        L_sc = self.params["L_sc"]

        # Gaussian barriers at both interfaces
        V_left = Z * mu * np.exp(-((x - L_normal) ** 2) / (2 * a**2))
        V_right = Z * mu * np.exp(-((x - (L_normal + L_sc)) ** 2) / (2 * a**2))

        return V_left + V_right

    def build_system(self):
        """Build the FM/SC/FM junction system."""
        dx = self.params["dx"]
        L_normal = self.params["L_normal"]
        L_sc = self.params["L_sc"]

        # Define 1D lattice with 2 orbitals per site
        lat = kwant.lattice.chain(dx, norbs=2)

        # Create system
        system = kwant.Builder()

        # Define onsite and hopping functions
        def onsite(site):
            return self.make_onsite(
                site, self.h_function, self.delta_function, self.V_function
            )

        def hopping(site1, site2):
            return self.make_hopping(site1, site2, self.h_function, self.delta_function)

        # Add sites and hoppings to the system
        # Left ferromagnet region
        left_fm_sites = [lat(i) for i in range(int(L_normal / dx))]
        for site in left_fm_sites:
            system[site] = onsite

        # Superconductor region
        sc_sites = [
            lat(i) for i in range(int(L_normal / dx), int((L_normal + L_sc) / dx))
        ]
        for site in sc_sites:
            system[site] = onsite

        # Right ferromagnet region
        right_fm_sites = [
            lat(i)
            for i in range(int((L_normal + L_sc) / dx), int((2 * L_normal + L_sc) / dx))
        ]
        for site in right_fm_sites:
            system[site] = onsite

        # Add hoppings between all adjacent sites
        all_sites = left_fm_sites + sc_sites + right_fm_sites
        for i in range(len(all_sites) - 1):
            system[all_sites[i], all_sites[i + 1]] = hopping

        # Add left lead (ferromagnet)
        sym_left = kwant.TranslationalSymmetry((-dx,))
        left_lead = kwant.Builder(sym_left, conservation_law=self.sigma_law)

        left_lead[lat(0)] = onsite
        left_lead[lat(0), lat(1)] = hopping

        # Add right lead (ferromagnet)
        sym_right = kwant.TranslationalSymmetry((dx,))
        right_lead = kwant.Builder(sym_right, conservation_law=self.sigma_law)

        right_lead[lat(0)] = onsite
        right_lead[lat(0), lat(1)] = hopping

        # Attach leads
        system.attach_lead(left_lead)
        system.attach_lead(right_lead)

        # Finalize the system
        self.system = system.finalized()
        return self.system

    def calculate_coefficients(self, energy):
        """Calculate reflection and transmission coefficients at given energy."""
        if self.system is None:
            self.build_system()

        # Calculate scattering matrix
        smatrix = kwant.smatrix(self.system, energy)

        # Calculate reflection probabilities
        # (0,0) -> electron from left lead to electron in left lead (normal reflection)
        Ree = smatrix.transmission((0, 0), (0, 0))

        # (0,1) -> electron from left lead to hole in left lead (Andreev reflection)
        Rhe = smatrix.transmission((0, 1), (0, 0))

        # Calculate transmission probabilities
        # (1,0) -> electron from left lead to electron in right lead
        Tee = smatrix.transmission((1, 0), (0, 0))

        # (1,1) -> electron from left lead to hole in right lead (crossed Andreev reflection)
        The = smatrix.transmission((1, 1), (0, 0))

        return Ree, Rhe, Tee, The

    def run_simulation(self):
        """Run the simulation for FM/SC/FM junction."""
        results = {}

        # Build the system
        self.build_system()

        # Task 1: Calculate coefficients vs energy for different SC lengths
        energy_min, energy_max = self.params["energy_range"]
        energies = np.linspace(energy_min, energy_max, 100)

        # For different SC lengths
        L_sc_values = [10, 250]  # nm

        for L_sc in L_sc_values:
            self.params["L_sc"] = L_sc
            self.build_system()

            Ree_vs_E = []
            Rhe_vs_E = []
            Tee_vs_E = []
            The_vs_E = []

            for E in energies:
                Ree, Rhe, Tee, The = self.calculate_coefficients(E)
                Ree_vs_E.append(Ree)
                Rhe_vs_E.append(Rhe)
                Tee_vs_E.append(Tee)
                The_vs_E.append(The)

            results[f"L_sc_{L_sc}"] = {
                "energies": energies,
                "Ree": Ree_vs_E,
                "Rhe": Rhe_vs_E,
                "Tee": Tee_vs_E,
                "The": The_vs_E,
            }

        # Task 2: Calculate coefficients vs SC length for fixed energy
        E_fixed = 0.1  # meV
        L_sc_range = np.linspace(1, 250, 50)  # nm

        # Case 1: P_left = P_right = 0
        self.params["P_left"] = 0.0
        self.params["P_right"] = 0.0

        Ree_vs_L = []
        Rhe_vs_L = []
        Tee_vs_L = []
        The_vs_L = []

        for L_sc in L_sc_range:
            self.params["L_sc"] = L_sc
            self.build_system()

            Ree, Rhe, Tee, The = self.calculate_coefficients(E_fixed)
            Ree_vs_L.append(Ree)
            Rhe_vs_L.append(Rhe)
            Tee_vs_L.append(Tee)
            The_vs_L.append(The)

        results["L_sc_sweep_P0"] = {
            "L_sc_values": L_sc_range,
            "Ree": Ree_vs_L,
            "Rhe": Rhe_vs_L,
            "Tee": Tee_vs_L,
            "The": The_vs_L,
        }

        # Case 2: P_left = 0.995, P_right = 0
        self.params["P_left"] = 0.995
        self.params["P_right"] = 0.0

        Ree_vs_L = []
        Rhe_vs_L = []
        Tee_vs_L = []
        The_vs_L = []

        for L_sc in L_sc_range:
            self.params["L_sc"] = L_sc
            self.build_system()

            Ree, Rhe, Tee, The = self.calculate_coefficients(E_fixed)
            Ree_vs_L.append(Ree)
            Rhe_vs_L.append(Rhe)
            Tee_vs_L.append(Tee)
            The_vs_L.append(The)

        results["L_sc_sweep_P995"] = {
            "L_sc_values": L_sc_range,
            "Ree": Ree_vs_L,
            "Rhe": Rhe_vs_L,
            "Tee": Tee_vs_L,
            "The": The_vs_L,
        }

        return results


class TaskManager:
    """Manager class for running KWANT tasks independently."""

    def __init__(self):
        """Initialize task manager."""
        self.tasks = {}
        self.results = {}

    def register_task(self, task_instance):
        """Register a task with the manager."""
        task_name = task_instance.name
        self.tasks[task_name] = task_instance
        return task_name

    def run_task(self, task_name):
        """Run a specific task by name."""
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not registered")

        task = self.tasks[task_name]
        print(f"Running task: {task_name}")

        # Run the task and store results
        self.results[task_name] = task.run_simulation()

        print(f"Task '{task_name}' completed")
        return self.results[task_name]

    def run_all_tasks(self):
        """Run all registered tasks."""
        for task_name in self.tasks:
            self.run_task(task_name)

        return self.results

    def get_task_result(self, task_name):
        """Get results for a specific task."""
        if task_name not in self.results:
            raise ValueError(f"No results for task '{task_name}'")

        return self.results[task_name]

    def get_all_results(self):
        """Get all task results."""
        return self.results


# Example usage
if __name__ == "__main__":
    # Create task manager
    manager = TaskManager()

    # Create and register tasks
    task1 = NormalSuperconductorTask()
    task2 = FerromagnetSuperconductorFerromagnetTask()

    manager.register_task(task1)
    manager.register_task(task2)

    # Run specific task
    # results = manager.run_task(task1.name)

    # Or run all tasks
    all_results = manager.run_all_tasks()

    print("Tasks are ready to run. Use the TaskManager to execute them.")
