from typing import List, Optional
from ax.modelbridge.factory import get_sobol
from sklearn.preprocessing import normalize
from boppf.utils.ax import get_parameters
from ax.service.ax_client import AxClient
from boppf.utils.data import frac_names
import plotly.express as px
from plotly import offline

subfrac_names, parameters = get_parameters(remove_composition_degeneracy=False)

# quick way of making a search_space
use_constraint = True
if use_constraint:
    comp_constraints: Optional[List] = [
        f"{subfrac_names[0]} + {subfrac_names[1]} <= 1.0"
    ]
    
else:
    comp_constraints = None
ax_client = AxClient()
ax_client.create_experiment(
    parameters=parameters, parameter_constraints=comp_constraints
)
search_space = ax_client.experiment.search_space

m = get_sobol(search_space, seed=10, fallback_to_sample_polytope=True)

# actual param_df may not match number of rows requested exactly
gr = m.gen(n=2 ** 10)  # 2**10==1024, 2**11==2048
df = gr.param_df

# normalize the compositional parameters
# (workaround for Sobol sampling bias, see https://github.com/facebook/Ax/issues/903)
# not sure if random sampling would be better
# df[frac_names] = normalize(df[frac_names], norm="l1")

# check the density of the scatter points
fig = px.scatter_3d(df[frac_names], x=frac_names[0], y=frac_names[1], z=frac_names[2])
fig.update_traces(marker_size=5)
offline.plot(fig)
1 + 1

