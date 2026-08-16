# RAOC model code

Kinetic model of sulfamethazine (SMT) degradation in a rotating advanced
oxidation contactor (RAOC), together with its dimensionless reduction and an
ecotoxicity extension.

The RAOC is a horizontal drum coated with TiO₂/zeolite composite sheets,
half-submerged in the bulk liquid and irradiated over the exposed arc. A fixed
point on the sheet alternates between a submerged phase, where solute is
exchanged with the bulk by convective mass transfer, and an exposed phase,
where the liquid film it carries is irradiated and photocatalytic degradation
proceeds.

## Files

| File | Contents |
|---|---|
| `raoc_simulator.py` | Dimensional forward model. Parameters, mass-transfer and film correlations, the eight-equation ODE system, and the experimental design matrix. |
| `raoc_dimensionless.py` | Dimensionless reduction of the same model to Π-groups, plus an equivalence check against `raoc_simulator.py`. |
| `ecotoxicity_layer.py` | Extension. Resolves the lumped intermediate pools into individual transformation products and converts them to predicted algal growth inhibition. |

`raoc_dimensionless.py` and `ecotoxicity_layer.py` both read
`raoc_simulator.py`; no parameter values are duplicated between files.

## Requirements

Python 3.12 with `numpy`, `scipy` and `pandas`. Tested on numpy 2.4.4,
scipy 1.17.1, pandas 3.0.2.

```
pip install numpy scipy pandas
```

## Running

Each file runs standalone from the directory holding all three.

```bash
python raoc_simulator.py       # observed rate constants for the 14 conditions
python raoc_dimensionless.py   # Π-groups and the equivalence check
python ecotoxicity_layer.py    # speciation and predicted inhibition
```

`raoc_dimensionless.py` looks for `raoc_simulator.py` alongside itself. To
point it elsewhere, set `RAOC_DIM_PATH`:

```bash
RAOC_DIM_PATH=/path/to/raoc_simulator.py python raoc_dimensionless.py
```

## Use as a library

```python
import raoc_simulator as sim

# one operating condition
rec = sim.simulate_condition(rpm=20, T_C=25, I_mW=1.0,
                             A_cm2=800, V_bulk_L=2.0, C0=10.0)
rec['Cb_P']        # bulk parent concentration at the sample times [mg/L]
rec['k_obs_init']  # initial first-order rate constant [1/h]

# the full design matrix
results = sim.run_conditions()
```

```python
import raoc_dimensionless as nd

df, s = nd.simulate(rpm=20., T_C=25., I_mW=1., A_cm2=800.,
                    V_bulk_L=2., C0=10.)
nd.print_groups(s)   # Π-group inventory for this condition
df['t_tilde']        # dimensionless time
```

```python
import ecotoxicity_layer as tox

df = tox.compute_toxicity(tox.records_to_frame(sim.run_conditions()))
df['TU_total_neutral']                          # toxic units
tox.TU_to_inhibition(df['TU_total_neutral'])    # inhibition, fraction of control
```
