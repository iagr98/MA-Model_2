import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from test_script import run_sim

N_CPU = 4
N_d = [30, 50, 80, 100, 150, 200, 300, 400]
var = 'N_d'                                                             # Update


def parallel_simulation(params):
    N_d = params                                                        # Update parameters
    print(f"Start simulation with {var}={N_d}")                         # Update parameter in second {}
    try:
        Sim = run_sim(N_s=N_d)                                          # Update inputs
        return {f"{var}": N_d,                                          # Update parameter in second place
                'V_dis_total': Sim.V_dp, 'Sep. Eff.': Sim.E,
                'status': 'success'}    
    except Exception as e:
        print(f"Simulation failed by {var}={N_d}: {str(e)}")            # Update parameter in second {}
        return {f"{var}": N_d, 'error': str(e), 'status': 'failed'}     # Update parameter in second place

if __name__ == "__main__":
    parameters = [N_d_value for N_d_value in N_d]                       # Update parameter var_value, var_value & var 
    
    results = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel_N_d.csv', index=False)   # update name of file
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")

    # Plot results
    df = pd.read_csv("simulation_results_parallel_N_d.csv")                 # update name of file
    df.columns = df.columns.str.strip()
    plt.figure(figsize=(8, 5))
    plt.plot(df['N_d'], df['V_dis_total'], marker='o')                # Update parameter in first place
    # plt.xscale('log')  # da atol logarithmisch skaliert ist
    # plt.yscale('log')  # da atol logarithmisch skaliert ist
    plt.xlabel('N_d')                                                   # Change x-label
    plt.ylabel('V_dp')                                             # Change output variable if needed
    plt.title(f'Gitterunabhängigkeitsanalyse ({var})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(df['N_d'], df['Sep. Eff.'], marker='o')                    # Update parameter in first place
    plt.ylabel('Sep. Eff.')                                             # Change output variable if needed
    plt.xlabel('N_d')                                                   # Change x-label
    plt.title(f'Gitterunabhängigkeitsanalyse ({var})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()