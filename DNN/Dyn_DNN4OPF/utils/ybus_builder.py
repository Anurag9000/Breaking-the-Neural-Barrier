"""
ybus_builder.py
===============

Utility for constructing a sparse complex bus–admittance (Y-bus) matrix
from the JSON representation used by the synthetic PGLib samples.

The function returns four 1-D tensors that can be packed into
`opf_data["y_bus"]` exactly as expected by
`constraint_losses.compute_all_constraint_losses(...)`.
"""

import torch
import math
import numpy as np

def construct_admittance_matrix(original_data: dict):
    """Restructures raw grid data and constructs admittance matrix."""

    # --- Data Restructuring Section ---
    data = {
        'grid': {
            'context': [original_data['grid']['context'][0][0][0]],  # Extract base MVA
            'nodes': {
                'bus': {str(i): bus for i, bus in enumerate(original_data['grid']['nodes']['bus'])},
                'shunt': original_data['grid']['nodes']['shunt']
            },
            'edges': {
                'shunt_link': original_data['grid']['edges']['shunt_link']
            }
        },
        'grids': {
            'edges': {}
        },
        'elements': {
            'shunt': {}
        }
    }

    # Process AC lines
    ac_line = original_data['grid']['edges']['ac_line']
    for idx, (f, t) in enumerate(zip(ac_line['senders'], ac_line['receivers'])):
        data['grids']['edges'][str(idx)] = {
            'in_service': True,
            'from_bus': str(f),
            'to_bus': str(t),
            'resistance': ac_line['features'][idx][4],  # br_r
            'reactance': ac_line['features'][idx][5],   # br_x
            'charging_susceptance': ac_line['features'][idx][2],  # b_fr
            'branch_type': 'ac_line',
            'rating_long_term': ac_line['features'][idx][6]  # rate_a
        }

    # Process transformers
    transformer = original_data['grid']['edges']['transformer']
    start_idx = len(data['grids']['edges'])
    for idx, (f, t) in enumerate(zip(transformer['senders'], transformer['receivers'])):
        data['grids']['edges'][str(start_idx + idx)] = {
            'in_service': True,
            'from_bus': str(f),
            'to_bus': str(t),
            'resistance': transformer['features'][idx][2],  # br_r
            'reactance': transformer['features'][idx][3],   # br_x
            'charging_susceptance': transformer['features'][idx][9],  # b_fr
            'branch_type': 'transformer',
            'transformer_tap_ratio': transformer['features'][idx][7],  # tap
            'transformer_phase_shift': transformer['features'][idx][8],  # shift
            'rating_long_term': transformer['features'][idx][4]  # rate_a
        }
    
    # Process shunts
    for idx, (bs, gs) in enumerate(original_data['grid']['nodes']['shunt']):
        data['elements']['shunt'][str(idx)] = {
            'bus': str(original_data['grid']['edges']['shunt_link']['receivers'][idx]),
            'bs': bs,
            'gs': gs
        }

    mva_base = data['grid']['context'][0]

    buses = [(key, val) for key, val in data['grid']['nodes']['bus'].items()]
    buses.sort(key=lambda x: int(x[0]))

    branches = [(key, val) for key, val in data['grids']['edges'].items()
                if val['in_service']]
    branches.sort(key=lambda x: int(x[0]))

    idx_to_bus = [key for (key, _) in buses]
    bus_to_idx = {x[0]: i for (i, x) in enumerate(buses)}

    idx_to_branch = [key for (key, _) in branches]
    branch_to_idx = {x[0]: i for (i, x) in enumerate(branches)}

    branch_matrix_list = []
    I, J, V = [], [], []

    for _, branch in branches:
        if not branch['in_service']:
            continue

        from_bus = branch['from_bus']
        to_bus = branch['to_bus']
        if from_bus not in bus_to_idx or to_bus not in bus_to_idx:
            continue

        f_bus = bus_to_idx[from_bus]
        t_bus = bus_to_idx[to_bus]

        rs = branch['resistance']
        xs = branch['reactance']
        bs = branch['charging_susceptance']

        # --------- NEW: guard against zero-impedance divisions --------------
        eps = 1e-6
        if abs(rs) < eps and abs(xs) < eps:
            rs = eps
        # --------------------------------------------------------------------

        y = 1/(rs + xs * 1j)
        lc_fr = bs * 1j
        lc_to = bs * 1j

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = branch['transformer_phase_shift']

        tr = tau * math.cos(math.radians(shift))
        ti = tau * math.sin(math.radians(shift))
        t = tr + ti * 1j

        a = (y + lc_fr) / (t.real ** 2 + t.imag ** 2)
        b = -y / np.conjugate(t)
        c = -y / t
        d = y + lc_to

        I += [f_bus, t_bus, f_bus, t_bus]
        J += [t_bus, f_bus, f_bus, t_bus]
        V += [b, c, a, d]

        Y_branch = torch.tensor([[a, b], [c, d]], dtype=torch.cfloat)
        thermal_limit = branch['rating_long_term'] / mva_base

        branch_matrix_list.append({
            'admittance_matrix': torch.tensor([[a, b], [c, d]], dtype=torch.complex64),
            'bus': (from_bus, to_bus),
            'idx': (f_bus, t_bus),
            'thermal_limit': torch.tensor(branch['rating_long_term']/mva_base)
        })

    # Add shunt elements
    for _, shunt in data['elements']['shunt'].items():
        shunt_bus = shunt['bus']
        if shunt_bus in bus_to_idx:
            bus = bus_to_idx[shunt_bus]
            ys = torch.complex(
                torch.tensor(shunt['gs']),
                torch.tensor(shunt['bs'])
            )
            I.append(bus)
            J.append(bus)
            V.append(ys)

    # Create sparse tensor
    indices = torch.tensor([I, J], dtype=torch.long)
    values = torch.tensor(V, dtype=torch.complex64)

    # --------- NEW: strip any lingering nan/inf just in case ---------------
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    # -----------------------------------------------------------------------

    shape = (len(buses), len(buses))
    y_bus = torch.sparse_coo_tensor(indices=indices, values=values, size=shape).coalesce()

    # Extract final components from y_bus
    coalesced_values = y_bus.values()
    coalesced_indices = y_bus.indices()
    coalesced_shape = y_bus.size()

    return coalesced_values, coalesced_indices[0], coalesced_indices[1], coalesced_shape

# with open("data\sample_case14.json", "r", encoding="utf-8") as f:
#         raw_json = json.load(f)

# s=construct_admittance_matrix(raw_json)
# print(s)
