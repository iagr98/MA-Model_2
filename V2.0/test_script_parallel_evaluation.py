import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from test_script import run_sim

N_CPU = 8

experiment = "detail_V_dis" # "main" for ye + niba experiements, "sozh" for experiments from AVT.FVT lab.

df = pd.read_excel("settings/data_main.xlsx", sheet_name=experiment)
exp = df['exp'].tolist()
phi_0 = df['phi_0'].tolist()
dV_ges = df['dV_ges'].tolist()
eps_0 = df['eps_0'].tolist()
if (experiment == "sozh" or experiment == "detail_V_dis"):
    h_dp_0 = df['h_dis_0'].tolist()
    h_w_0 = df['h_c_0'].tolist()   # CHANGE for option 1 or 2.


def parallel_simulation(params):
    if (experiment == "main" or experiment == "detail_lambda"):
        exp, phi_0, dV_ges, eps_0 = params
    elif (experiment == "sozh" or experiment == "detail_V_dis"):
        exp, phi_0, dV_ges, eps_0, h_dp_0, h_w_0 = params
    print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}")
    try:
        if (experiment == "main" or experiment == "detail_lambda"):
            Sim = run_sim(exp, phi_0, dV_ges, eps_0)
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                    'V_dp': Sim.V_dp, 'Sep. Eff.': Sim.E,'h_dpz': Sim.h_dpz ,'L_dp': Sim.L_dp , 'status': 'success'}
        elif (experiment == "sozh" or experiment == "detail_V_dis"):
            Sim = run_sim(exp, phi_0, dV_ges, eps_0, h_dp_0, h_w_0)
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                    'h_dp_0': h_dp_0, 'h_w_0': h_w_0,
                    'V_dp': Sim.V_dp, 'Sep. Eff.': Sim.E,'h_dpz': Sim.h_dpz ,'L_dp': Sim.L_dp , 'status': 'success'}
    except Exception as e:
        if (experiment == "main" or experiment == "detail_lambda"):
            print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}: {str(e)}")
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'error': str(e), 'status': 'failed'}
        elif (experiment == "sozh" or experiment == "detail_V_dis"):
            print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, h_dp_0={h_dp_0}, h_w_0={h_w_0}: {str(e)}")
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 
                    'h_dp_0': h_dp_0, 'h_w_0': h_w_0, 'error': str(e), 'status': 'failed'}

if __name__ == "__main__":
    
    if (experiment == "main" or experiment == "detail_lambda"):
        parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i]) for i in range(len(exp))]
    elif (experiment == "sozh" or experiment == "detail_V_dis"):
        parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i], h_dp_0[i], h_w_0[i]) for i in range(len(exp))]
    
    results = joblib.Parallel(n_jobs=N_CPU, backend='multiprocessing')(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    h_dpz_columns = pd.DataFrame(df_results['h_dpz'].tolist())   # Convert h_dpz (list of arrays) into separate columns
    h_dpz_columns.columns = [f'h_dpz_{i}' for i in range(h_dpz_columns.shape[1])]
    df_results = df_results.drop(columns=['h_dpz'])
    df_results = pd.concat([df_results, h_dpz_columns], axis=1)  # Concatenate V_dis columns with the main result dataframe
    df_results.to_csv('simulation_results_parallel_evaluation_detail_final.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")

   