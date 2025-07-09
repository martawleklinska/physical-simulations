import numpy as np
import matplotlib.pyplot as plt
from manus_test import (
    NormalSuperconductorTask,
    FerromagnetSuperconductorFerromagnetTask,
    TaskManager,
)


def test_task_independence():
    """Test that tasks can be run independently through the task manager."""
    print("Testing task independence...")

    # Create task manager
    manager = TaskManager()

    # Create tasks with minimal parameters for quick testing
    task1_params = {
        "L_normal": 50,  # Shorter length for faster testing
        "L_sc": 50,
        "energy_range": [0, 0.3],
    }

    task2_params = {"L_normal": 50, "L_sc": 10, "energy_range": [0, 0.3]}

    # Create and register tasks
    task1 = NormalSuperconductorTask(task1_params)
    task2 = FerromagnetSuperconductorFerromagnetTask(task2_params)

    task1_name = manager.register_task(task1)
    task2_name = manager.register_task(task2)

    print(f"Registered tasks: {list(manager.tasks.keys())}")

    # Test running tasks independently
    print("\nRunning first task independently...")
    results1 = manager.run_task(task1_name)

    print("\nRunning second task independently...")
    results2 = manager.run_task(task2_name)

    # Verify results structure
    print("\nVerifying results structure...")
    assert "Z_study" in results1, "Missing Z_study in task1 results"
    assert "P_study" in results1, "Missing P_study in task1 results"
    assert "L_sc_10" in results2, "Missing L_sc_10 in task2 results"
    assert "L_sc_sweep_P0" in results2, "Missing L_sc_sweep_P0 in task2 results"

    print("All tests passed! Tasks can be run independently.")
    return True


if __name__ == "__main__":
    test_task_independence()
