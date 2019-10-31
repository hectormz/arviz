import arviz as az
data = az.load_arviz_data('non_centered_eight')
coords = {'school': ['Choate', 'Lawrenceville']}
az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords)
