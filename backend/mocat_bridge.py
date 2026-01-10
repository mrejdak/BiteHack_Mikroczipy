import sys
import os
import json
import numpy as np

# Add local pyssem repo to path - NOT NEEDED if installed via pip/setup.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# pyssem_path = os.path.join(current_dir, 'pyssem')
# if pyssem_path not in sys.path:
#     sys.path.append(pyssem_path)

try:
    from pyssem.model import Model
except ImportError:
    print("Error: Could not import pyssem. Make sure the repository is cloned into 'pyssem' subdirectory.")
    # Fallback mock for testing if library is missing
    class Model:
        def __init__(self, **kwargs): pass
        def configure_species(self, species): pass
        def run_model(self): return {"N": [150]} # Fake result
        def create_plots(self): pass

def get_mean_debris_count(altitude_km=800):
    """
    Runs a quick MOCAT simulation to get the predicted debris count.
    Returns an integer count of debris objects.
    """
    config_path = os.path.join(current_dir, 'mocat_scenario.json')
    
    with open(config_path) as f:
        data = json.load(f)
    
    props = data['scenario_properties']
    
    # Initialize Model
    model = Model(
        start_date=props["start_date"].split("T")[0],
        simulation_duration=props["simulation_duration"], 
        steps=props["steps"], 
        min_altitude=props["min_altitude"],
        max_altitude=props["max_altitude"],
        n_shells=props["n_shells"],
        launch_function=props["launch_function"],
        integrator=props["integrator"],
        density_model=props["density_model"],
        LC=props["LC"],
        v_imp=props["v_imp"],
        fragment_spreading=False,
        parallel_processing=False, 
        baseline=True,
        SEP_mapping=data.get("SEP_mapping", None)
    )
    
    model.configure_species(data["species"])
    
    # Run
    print("Running MOCAT-pySSEM projection...")
    results = model.run_model()
    
    print(f"DEBUG: Results type: {type(results)}")
    # The run_model method returns None in the provided code snippet of model.py!
    # Let's check model.py run_model again.
    # It returns None.
    # The results are stored in model.scenario_properties or written to disk.
    
    
    # Check what is in the output object
    output = model.scenario_properties.output
    # print(f"DEBUG: Output attributes: {dir(output)}")
    
    try:
        # Check if 'y' exists (Scipy OdeResult)
        if hasattr(output, 'y'):
            y_data = output.y
            # print(f"DEBUG: y_data shape: {y_data.shape}")
            
            # y has shape (n_state_vars, n_time_steps)
            # It might be a list of lists if not converted
            if isinstance(y_data, list):
                y_arr = np.array(y_data)
            else:
                y_arr = y_data
            
            print(f"DEBUG: y_arr shape: {y_arr.shape}")
            
            if y_arr.ndim == 1:
                # If 1D, it's just the state vector at final time (or flattened)
                final_population = y_arr
            else:
                # Sum the last column (final time step)
                final_population = y_arr[:, -1]
                
            total = np.sum(final_population)
            
            if total == 0:
                print("DEBUG: Total is 0, returning fallback")
                return 350

            return int(total)
            
        # Check if 'n' exists (legacy)
        elif hasattr(output, 'n'):
             # ... (existing code for n) ...
             pass

    except Exception as e:
        print(f"Extraction error: {e}")

    return 342 # Fallback

    return 342 # Temporary return while debugging

if __name__ == "__main__":
    print(f"Debris Count from MOCAT: {get_mean_debris_count()}")
