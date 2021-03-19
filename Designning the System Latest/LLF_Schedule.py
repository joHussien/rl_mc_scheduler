from env.job_generator import create_workload
import numpy as np


def compute_SCP_MBS(num_jobs, total_load, lo_per, job_density, time):
        # We schedule here using LLF and compute the MBS of the generated Schedule
        workload = create_workload(num_jobs, total_load, lo_per, job_density)
        print("Generated Workload")
        print(workload)
        print("LLF-Schedule")
        lax = np.subtract(workload[:, 1], workload[:, 2])
        min_critical_points = np.argsort(lax, axis=0)
        print(min_critical_points)
        chosen=workload[min_critical_points]
        print(chosen)
        Hi_index=np.where(chosen[:,3]==1)
        chosen_Hi=chosen[Hi_index]
        total_processing = np.sum(chosen_Hi[2], axis=0)
        MBS = total_processing / max(workload[:][1])

        print("MBS of this Schedule: ", 1 / MBS)



compute_SCP_MBS(10, 0.5, 0.3, 5, 0)

