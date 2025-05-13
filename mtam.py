import numpy as np
import pickle
import graph_tool.all as gt

class mtam():
    def __init__(self, *, deg_corr=True, nested=True):
        self.g = None
        self.trope_elements = []
        self.manuscripts = []
        self.states = []
        self.groups = {}
        self.mdl = np.nan
        self.mdl_history = []

        # Inference settings
        self.deg_corr = deg_corr
        self.nested = nested


    def load_model(self, path):
        if ".pickle" not in path:
            path += ".pickle"
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)

    def refine_states(self):
        best_mdl = np.inf
        n_init = len(self.states)
        for i in range(n_init):
            state_tmp = self.states[i]["state"]
            print("Run ", i)
            print("beta = 1")
            for _ in range(1000):
                state_tmp.multiflip_mcmc_sweep(beta=1, niter=10)
                cur_mdl = state_tmp.entropy()
                self.mdl_history[i].append(cur_mdl)
                if cur_mdl < best_mdl:
                    print(cur_mdl)
                    best_mdl = cur_mdl
                    best_state = state_tmp.copy()

            print("beta = inf")
            for _ in range(2000):
                state_tmp.multiflip_mcmc_sweep(beta=np.inf, niter=10)
                cur_mdl = state_tmp.entropy()
                self.mdl_history[i].append(cur_mdl)
                if cur_mdl < best_mdl:
                    print(cur_mdl)
                    best_mdl = cur_mdl
                    best_state = state_tmp.copy()

            print("Finished")
            self.states.append({
                "iteration": i,
                "state": best_state,
                "mdl": best_state.entropy(),
                "levels": best_state.get_levels()
            })

        self.mdl = min(state["mdl"] for state in self.states)
