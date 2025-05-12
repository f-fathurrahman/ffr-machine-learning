import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color": "gray"
})

pd.options.mode.chained_assignment = None
