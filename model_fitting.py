"""
Biological Circuit Characterization Pipeline

This script performs model fitting and analysis for induce OA circuit experimental data.
Includes dose-response curve fitting, optical density correction, and parameter optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import pickle
import os
import logging
from typing import NamedTuple, Tuple, Dict, List
from scipy.optimize import curve_fit, least_squares, minimize
from scipy import stats
from matplotlib.patches import PathPatch

# Configuration constants
CONFIG = {
    'n': 3,
    'k0': 2000,
    'beta': 1.0,
    'min_expression': 0.001,
    'fig_dir': './output/Figure/',
    'input_paths': {
        'induction_rpu': './input/induction_RPU/',
        'od_data': './input/induction_OD/',
        'system_params': './input/induce_system_parameters.csv',
        'reporter_list': './input/reporter_list.csv'
    },
    'plot_defaults': {
        'figsize': (6, 6),
        'dpi': 300,
        'color': 'red',
        'marker_size': 20
    },
    'fit_bounds': {
        'hill': ([-np.inf, -np.inf, 2-1e-7], [np.inf, np.inf, 4]),
        'od_params': ([0, -0.2, 0, -0.2], [10, 0, 10, 0])
    }
}

INDUCE_CIRCUIT_LIST = [
            'cwj458+629-MG1655_20230404',
            'cwj458+629-MG1655_20230405', 
            'cwj458+629-MG1655_20230406',
            'cwj460+630-MG1655_202305-3',
            'cwj460+630-MG1655_202305-1',
            'cwj460+630-MG1655_202305-2',
            'cwj582+552-MG1655_20230405',
            'cwj582+552-MG1655_202304072',
            'cwj582+552-MG1655_20230407',
            'cwj582+715-BL21_20230530',
            'cwj583+620-MG1655_20230504',
            'cwj583+680-BL21_20230526',
            'cwj583+680-BL21_20230527',
            'cwj583+680-BL21_20230528',
            'cwj719+552-MG1655_20230617',
            'cwj719+552-MG1655_20231101',
            'cwj719+552-MG1655_20231102',
            'cwj584+715-MG1655_20240111',
            'cwj584+715-MG1655_20240113',
            'cwj584+715-MG1655_20240117',
            'cwj438+1115-MG1655_20240722',
            'cwj438+1115-MG1655_20240723',
            'cwj438+1115-MG1655_20240724',
            'cwj1230+715-BL21_20241116-1',
            'cwj1230+715-BL21_20241122',
            'cwj1230+715-BL21_20241128',
            'cwj1232+715-BL21_20241116-1',
            'cwj1232+715-BL21_20241128',
            'cwj1232+715-BL21_20241129',
            'cwj793+714-MG1655_202410131',
            'cwj793+714-MG1655_202410132',
            'cwj793+714-MG1655_202410133',
            'cwj796+716-MG1655_20241015',
            'cwj796+716-MG1655_202410152',
            'cwj796+716-MG1655_202410153',
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelParameters(NamedTuple):
    """Container for model fitting parameters"""
    alpha: float = 0.0
    beta: float = 1.0
    k1: float = 0.0
    k2: float = 0.0
    ymax: float = 0.0
    r_squared: float = 0.0
    linear_r: float = 0.0

def hill_equation(x: np.ndarray, k: float, a: float, n: float) -> np.ndarray:
    """Hill equation for dose-response modeling"""
    return a * x**n / (x**n + k)

def plot_dose_response(concentrations: List[float], responses: List[float], name: str) -> np.ndarray:
    """Plot and fit dose-response curve using Hill equation"""
    try:
        popt, _ = curve_fit(
            hill_equation,
            concentrations,
            responses,
            p0=[2, 1, 3],
            bounds=CONFIG['fit_bounds']['hill']
        )
    except RuntimeError as e:
        logger.error(f"Curve fitting failed for {name}: {str(e)}")
        return np.array([])

    k, a, n = popt
    x_fit = np.linspace(0, math.ceil(max(concentrations)), 100)
    y_fit = hill_equation(x_fit, k, a, n)

    # Plotting logic
    fig, ax = plt.subplots(figsize=CONFIG['plot_defaults']['figsize'])
    ax.plot(x_fit, y_fit, color=CONFIG['plot_defaults']['color'],
           label=f'y = {a:.3f}x^{n:.3f}/(x^{n:.3f}+{k:.3f})')
    ax.scatter(concentrations, responses, label='Experimental Data')
    
    # Save figure
    output_dir = safe_mkdir(os.path.join(CONFIG['fig_dir'], 'inducer'))
    for ext in ['jpg', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'{name}-RPU.{ext}'))
    plt.close()

    return y_fit

def safe_mkdir(path: str) -> str:
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def load_rpu_data(filename: str) -> Tuple[List[List[float]], List[float]]:
    """Load RPU data from CSV file"""
    filepath = os.path.join(CONFIG['input_paths']['induction_rpu'], f'{filename}.csv')
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader if any(row)]

    # Data processing
    rpu_iptg = [max(float(val), 0) for val in data[1][2:]]
    rpu_ca = [max(float(row[1]), 0) for row in data[2:]]
    
    # Generate plots
    conc_iptg = select_concentration(len(rpu_iptg))
    plot_dose_response(conc_iptg, rpu_iptg, f'{filename}-IPTG')
    plot_dose_response([0, 2, 2.8, 3.5, 4.4, 5.8, 9.5, 50], rpu_ca, f'{filename}-CA')

    return process_input_pairs(data, rpu_ca, rpu_iptg)

def select_concentration(data_length: int) -> List[float]:
    """Select appropriate concentration series"""
    variants = {
        8: [0, 20, 27, 34, 42, 55, 76, 177],
        6: [0, 20, 34, 42, 76, 177]
    }
    return variants.get(data_length, [0])

def process_input_pairs(data: List[List[str]], rpu_ca: List[float], rpu_iptg: List[float]) -> Tuple[List[List[float]], List[float]]:
    """Process input pairs from raw data"""
    input_x = [[i, j] for j in rpu_ca for i in rpu_iptg]
    input_y = [max(float(row[i]), 0) for row in data[2:] for i in range(2, len(row))]
    return input_x, input_y[1:]

def main():
    """Main execution workflow"""
    parameter_model = {'MG1655': {}, 'BL21': {}}
    circuit_data = {}
    feature_data = {}
    results = []

    # Load reference data
    system_params = load_system_parameters()
    reporter_mapping = load_reporter_mapping()

    # Process circuit data
    for filename in INDUCE_CIRCUIT_LIST:
        process_circuit(filename, circuit_data, system_params, reporter_mapping)

    # Analyze and save results
    for cid, records in circuit_data.items():
        circuit_results = analyze_circuit(cid, records, parameter_model, feature_data)
        results.extend(circuit_results)
    
    save_results(results, parameter_model)

def load_system_parameters() -> Dict[str, List[str]]:
    """Load biological system parameters"""
    with open(CONFIG['input_paths']['system_params'], 'r') as f:
        return {row['ID']: [row['anti-sigma'], row['sigma']] for row in csv.DictReader(f)}

def load_reporter_mapping() -> Dict[str, str]:
    """Load reporter gene mapping"""
    with open(CONFIG['input_paths']['reporter_list'], 'r') as f:
        return {row['ID']: row['reporter'] for row in csv.DictReader(f)}

def process_circuit(filename: str, data_store: Dict, sys_params: Dict, reporters: Dict):
    """Process individual circuit data"""
    cid, date = parse_filename(filename)
    input_x, input_y = load_rpu_data(filename)
    od = load_od_measurements(filename)
    
    data_store.setdefault(cid, []).append([date, input_y, input_x, od])
    update_feature_data(cid, sys_params, reporters, data_store)

def parse_filename(filename: str) -> Tuple[str, str]:
    """Extract circuit ID and date from filename"""
    parts = filename.split('_')
    return parts[0], parts[-1]

def update_feature_data(cid: str, sys_params: Dict, reporters: Dict, data_store: Dict):
    """Update feature metadata for circuit"""
    if cid not in data_store:
        induce_id = cid.split('+')[0].replace('cwj', '')
        anti_sigma, sigma = sys_params.get(induce_id, ['', ''])
        reporter_id = extract_reporter_id(cid, reporters)
        
        data_store['feature'][cid] = (
            cid,
            cid.split('-')[-1],
            anti_sigma,
            sigma,
            reporter_id
        )

def analyze_circuit(cid: str, records: List, param_model: Dict, features: Dict) -> List:
    """Full analysis pipeline for a circuit"""
    aggregated = aggregate_data(records)
    
    try:
        model_params = fit_biological_model(
            responses=aggregated.y,
            inputs=aggregated.x,
            od_inputs=aggregated.od_x,
            od_measurements=aggregated.od_y,
            circuit_id=cid,
            features=features[cid]
        )
        
        if model_params.r_squared >= 0.9:
            param_model = update_parameter_model(cid, model_params, param_model, features)
            return format_output(cid, model_params, features[cid])
            
    except Exception as e:
        logger.error(f"Analysis failed for {cid}: {str(e)}")
    
    return []

def fit_biological_model(responses: List[float], inputs: List[List[float]], 
                        od_inputs: List, od_measurements: List, 
                        circuit_id: str, features: Tuple) -> ModelParameters:
    """Core model fitting with error handling"""
    X = np.array(inputs)
    y = np.array(responses)
    
    try:
        od_params = fit_od_model(od_inputs, od_measurements) if od_inputs else []
        fit_result = optimize_model(y, X, od_params)
        return validate_parameters(fit_result, features)
        
    except RuntimeError as e:
        logger.error(f"Model fitting failed for {circuit_id}: {str(e)}")
        return ModelParameters()

def optimize_model(y: np.ndarray, X: np.ndarray, od_params: List) -> Tuple:
    """Perform parameter optimization"""
    ymax = y.max()
    y_norm = y / ymax
    
    result = least_squares(
        error_function,
        [1.0, 1.0, 10.0],
        args=(X, y_norm),
        bounds=([0.1, 0.1, 1], [10.0, 10.0, 100.0])
    )
    
    alpha, k2, k1 = result.x
    r_squared = calculate_rsquared(y_norm, predict_output(X, alpha, k1, k2))
    
    return ModelParameters(alpha, CONFIG['beta'], k1, k2, ymax, r_squared, 0.0)

def predict_output(X: np.ndarray, alpha: float, k1: float, k2: float) -> np.ndarray:
    """Predict output using model parameters"""
    xe = calculate_xe(X, alpha, k1)
    return xe / (k2 + xe)

def calculate_xe(X: np.ndarray, alpha: float, k1: float) -> np.ndarray:
    """Calculate effective input"""
    A0 = alpha * np.maximum(X[:, 0], 0)
    B0 = CONFIG['beta'] * np.maximum(X[:, 1], 0)
    
    a = k1
    b = k1*(A0 + B0) + 1
    c = k1*A0*B0
    
    discriminant = np.sqrt(b**2 - 4*a*c)
    return np.maximum(A0 - (b - discriminant)/(2*a), 0)

def save_results(results: List, param_model: Dict):
    """Save final analysis results"""
    output_dir = safe_mkdir(CONFIG['fig_dir'])
    
    # Save CSV
    with open(os.path.join(output_dir, 'Model_Parameters.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'name', 'date', 'A_rbs', 'R_rbs', 
            'A_protein', 'R_protein', 'reporter', 
            'strain', 'ymax', 'alpha', 'beta', 
            'k1', 'k2', 'R2_standard', 'R2_linear'
        ])
        writer.writerows(results)
    
    # Save binary model
    with open(os.path.join(output_dir, 'parameter_model.pkl'), 'wb') as f:
        pickle.dump(param_model, f)

if __name__ == '__main__':
    main()
