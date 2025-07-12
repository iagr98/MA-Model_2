import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from test_script import run_sim

N_CPU = 8

experiment = "sozh" # "main" for ye + niba experiements, "sozh" for experiments from AVT.FVT lab.

df = pd.read_excel("settings/data_main.xlsx", sheet_name=experiment)
exp = df['exp'].tolist()
phi_0 = df['phi_0'].tolist()
dV_ges = df['dV_ges'].tolist()
eps_0 = df['eps_0'].tolist()
if experiment == "sozh":
    h_dp_max = df['h_dis_max'].tolist()
    h_w_0 = df['h_c_0'].tolist()   # CHANGE for option 1 or 2.


def parallel_simulation(params):
    if (experiment == "main"):
        exp, phi_0, dV_ges, eps_0 = params
    elif (experiment == "sozh"):
        exp, phi_0, dV_ges, eps_0, h_dp_max, h_w_0 = params
    print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}")
    try:
        if (experiment == "main"):
            Sim = run_sim(exp, phi_0, dV_ges, eps_0)
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                    'V_dp': Sim.V_dp, 'Sep. Eff.': Sim.E, 'status': 'success'}
        elif (experiment == "sozh"):
            Sim = run_sim(exp, phi_0, dV_ges, eps_0, h_dp_max, h_w_0)
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                    'h_dp_max': h_dp_max, 'h_w_0': h_w_0,
                    'V_dp': Sim.V_dp, 'Sep. Eff.': Sim.E, 'status': 'success'}
    except Exception as e:
        if (experiment == "main"):
            print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}: {str(e)}")
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'error': str(e), 'status': 'failed'}
        elif (experiment == "sozh"):
            print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, h_dp_max={h_dp_max}, h_w_0={h_w_0}: {str(e)}")
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 
                    'h_dp_max': h_dp_max, 'h_w_0': h_w_0, 'error': str(e), 'status': 'failed'}

if __name__ == "__main__":
    
    if (experiment == "main"):
        parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i]) for i in range(len(exp))]
    elif (experiment == "sozh"):
        parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i], h_dp_max[i], h_w_0[i]) for i in range(len(exp))]
    
    results = joblib.Parallel(n_jobs=N_CPU, backend='multiprocessing')(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel_evaluation_sozh.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")

   