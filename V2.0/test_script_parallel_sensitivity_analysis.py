import joblib
import pandas as pd
from test_script import run_sim

N_CPU = 8

dV_ges = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
phi_0 = [200e-6, 250e-6, 300e-6, 350e-6, 400e-6, 450e-6, 500e-6, 550e-6, 600e-6, 650e-6, 700e-6, 750e-6, 800e-6, 850e-6, 900e-6, 950e-6, 1000e-6]


def parallel_simulation(params):
    phi_0, dV_ges = params
    print(f"Start simulation with phi_0={phi_0}, dV_ges={dV_ges}")
    try:  
        Sim = run_sim(exp='sensitivity', phi_0=phi_0, dV_ges=dV_ges)
        return {'phi_0': phi_0, 'dV_ges': dV_ges, 'Sep. Eff.': Sim.E, 'status': 'success'}
        
    except Exception as e:
        print(f"Simulation failed for phi_0={phi_0}, dV_ges={dV_ges}: {str(e)}")
        return {'phi_0': phi_0, 'dV_ges': dV_ges,'error': str(e), 'status': 'failed'}

if __name__ == "__main__":
    
    parameters = [(phi, dV) for dV in dV_ges for phi in phi_0]
    
    results = joblib.Parallel(n_jobs=N_CPU, backend='multiprocessing')(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_sensitivity_lambda.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")