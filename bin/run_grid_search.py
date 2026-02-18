# This script evaluates UnCLe
import argparse
import os
import numpy as np
import time
from datetime import date
from experimental_utils import run_grid_search


def parse_sheet_name(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def parse_quantile_list(value):
    values = [v.strip() for v in str(value).split(",") if v.strip() != ""]
    if len(values) == 0:
        raise argparse.ArgumentTypeError("--binarize-quantiles cannot be empty")

    quantiles = []
    for v in values:
        try:
            q = float(v)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid quantile value: {v}") from exc

        if q < 0.0 or q > 1.0:
            raise argparse.ArgumentTypeError(f"Quantile must be within [0, 1], got: {v}")
        quantiles.append(q)

    return quantiles

parser = argparse.ArgumentParser(description='UnCLe Runner')


# Simulation model parameters
parser.add_argument('--experiment', type=str, default="lorenz96_0", help="Experiment to be performed (default: "
                                                                       "'lorenz96_0')")
parser.add_argument('--data-path', type=str, default=None,
                    help='Path to a custom dataset file (.csv/.xls/.xlsx).')
parser.add_argument('--structure-path', type=str, default=None,
                    help='Optional path to a ground-truth structure file (.csv/.xls/.xlsx).')
parser.add_argument('--sheet-name', type=parse_sheet_name, default=0,
                    help='Sheet name/index for Excel input files (default: 0).')
parser.add_argument('--results-only', action='store_true',
                    help='Only save inferred causal results, skip evaluation metrics.')
parser.add_argument('--binarize-quantile', type=str, default='0.9',
                    help='Quantile threshold for auto-binarization. Accepts a single value (e.g. 0.9) '
                         'or a comma-separated list (e.g. 0.8,0.9,0.95).')
parser.add_argument('--binarize-quantiles', type=parse_quantile_list, default=None,
                    help='Optional comma-separated quantiles for multi-threshold exports, '
                         'e.g. "0.8,0.9,0.95".')

# Model specification
parser.add_argument('--K', type=int, default=5, help='Kernel size (default: 5)')
parser.add_argument('--num-hidden-layers', type=int, default=1, help='Number of hidden layers (default: 1)')
parser.add_argument('--hidden-layer-size', type=int, default=50, help='Number of units in the hidden layer '
                                                                      '(default: 50)')

# Training procedure
parser.add_argument('--num-epochs-1', type=int, default=10, help='Number of epochs to train phase1 (default: 10)')
parser.add_argument('--num-epochs-2', type=int, default=10, help='Number of epochs to train phase2 (default: 10)')
parser.add_argument('--initial-lr', type=float, default=0.0001, help='Initial learning rate (default: 0.0001)')


# Meta
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
parser.add_argument('--num-sim', type=int, default=1, help='Number of simulations (default: 1)')
parser.add_argument('--use-cuda', type=bool, default=True, help='Use GPU? (default: true)')
parser.add_argument('--cuda-i', type=int, default=0, help='Cuda device number to use (default: 0)')



# Parsing args
args = parser.parse_args()

datasets = []
structures = []
signed_structures = None

print(str(args.num_sim) + " " + str(args.experiment) + " datasets...")

if args.experiment == "unicsl_lorenz96_0":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(f"../datasets/Lorenz96/Lorenz96_var20_force10_t250_data_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(f"../datasets/Lorenz96/Lorenz96_var20_force10_t250_struct_{i}.csv", index_col=None)
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_lorenz96_1":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(f"../datasets/Lorenz96/Lorenz96_var20_force40_t250_data_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(f"../datasets/Lorenz96/Lorenz96_var20_force40_t250_struct_{i}.csv", index_col=None)
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_lorenz96_2":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(f"../datasets/Lorenz96/Lorenz96_var100_force40_t500_data_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(f"../datasets/Lorenz96/Lorenz96_var100_force40_t500_struct_{i}.csv", index_col=None)
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_finance":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        if i % 2 != 0: continue
        data_i = pd.read_csv(f"../datasets/Finance/finance_data_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(f"../datasets/Finance/finance_struct_{i}.csv", index_col=None)
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_nc8":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(f"../datasets/NC8/nc8_data_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(f"../datasets/NC8/nc8_struct_{i}.csv", index_col=None)
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_fmri":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(f"../datasets/fMRI/fMRI_data_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(f"../datasets/fMRI/fMRI_struct_{i}.csv", index_col=None)
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_custom":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    if args.data_path is None:
        raise ValueError("--data-path is required when --experiment=unicsl_custom")

    ext = os.path.splitext(args.data_path)[1].lower()
    if ext in [".xls", ".xlsx"]:
        data_i = pd.read_excel(args.data_path, sheet_name=args.sheet_name)
    else:
        data_i = pd.read_csv(args.data_path, index_col=None)
    data_i[:] = StandardScaler().fit_transform(data_i[:])
    datasets.append(data_i.to_numpy())

    if args.structure_path is not None:
        struct_ext = os.path.splitext(args.structure_path)[1].lower()
        if struct_ext in [".xls", ".xlsx"]:
            a_i = pd.read_excel(args.structure_path, sheet_name=args.sheet_name)
        else:
            a_i = pd.read_csv(args.structure_path, index_col=None)
        structures.append(a_i.to_numpy())
    else:
        structures = None
else:
    raise NotImplementedError("ERROR: This experiment is not supported!")

compute_metrics = (not args.results_only) and (structures is not None)
binarize_quantiles = args.binarize_quantiles if args.binarize_quantiles is not None else parse_quantile_list(args.binarize_quantile)

run_grid_search(datasets=datasets, K=args.K, structures=structures,
                num_hidden_layers=args.num_hidden_layers, hidden_layer_size=args.hidden_layer_size,
                num_epochs_1=args.num_epochs_1, num_epochs_2=args.num_epochs_2, initial_lr=args.initial_lr,
                seed=args.seed, use_cuda=args.use_cuda,
                cuda_i=args.cuda_i, experiment_name=args.experiment, compute_metrics=compute_metrics,
                auto_binarize=True, binarize_quantiles=binarize_quantiles)
