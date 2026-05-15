# PINNS Mitchell-Schaeffer

This folder is for building a physics-informed neural network around the
Mitchell-Schaeffer cardiac reaction-diffusion model.

## Current Step

`ms_simulator_2d.py` generates reference data on a 2D rectangular tissue domain.
It is based on the equations used by `cardiac-simulator`, simplified to an
isotropic 2D finite-difference grid.

The simulated state variables are:

- `u`: scaled action potential
- `h`: recovery / gating variable

The model uses:

- no-flux boundaries
- focal stimulus in a circular patch
- smooth sigmoid approximation of the `h` threshold switch
- compressed `.npz` output for PINN training

Run:

```bash
python ms_simulator_2d.py
```

Outputs are written to `outputs/`:

- `ms_2d_data.npz`
- `ms_2d_snapshots.png`
- `ms_2d_centerline_xt.png`

## 2D PINN

`pinn_ms_2d_deepxde.py` trains a DeepXDE PINN using the generated 2D data.
It follows the same structure as `openCARP-PINNs`, but reads local `.npz`
arrays and uses a smooth sigmoid Mitchell-Schaeffer switch.

Install PINN dependencies:

```bash
pip install -r requirements-pinn.txt
```

Run a shorter first training pass:

```bash
python pinn_ms_2d_deepxde.py --adam-epochs 2000 --skip-lbfgs
```

Run the default training pass:

```bash
python pinn_ms_2d_deepxde.py
```

PINN outputs are written to `outputs/pinn_ms_2d/`.

## openCARP Setup

Install openCARP from the official Ubuntu `.deb`:

```text
https://opencarp.org/download/installation
```

Quick setup:

```bash
sudo apt-get update
sudo apt-get install git python3 python3-pip python3-testresources python-is-python3
cd ~/Downloads
sudo apt-get install ./opencarp-vXX.X.deb
openCARP -buildinfo
```

Set up `carputils`:

```bash
cd ~/github/UPenn_Py/piml_dt_cardiac
python3 -m venv cardiac_env
source cardiac_env/bin/activate
python -m pip install "setuptools<82" wheel
cd ~/github/UPenn_Py
git clone https://git.opencarp.org/openCARP/carputils.git
cd carputils
python -m pip install .
cusettings ~/.config/carputils/settings.yaml
cusummary
```

Run the basic 2D tissue example:

```bash
cp -r /usr/local/lib/opencarp/share/examples ~/opencarp-examples
cd ~/opencarp-examples/02_EP_tissue/01_basic_usage
source ~/github/UPenn_Py/piml_dt_cardiac/cardiac_env/bin/activate
python run.py --duration 100
find . -type f \( -name "*.igb" -o -name "*.pts" -o -name "*.elem" \)
```

Next, check the Mitchell-Schaeffer model/state variables:

```bash
bench --list-imps | grep -i schae
bench --imp MitchellSchaeffer --imp-info
```

The target openCARP dataset for PINN training is `vm.igb`, mesh `.pts`, and the
Mitchell-Schaeffer recovery variable dump, usually `h.igb`.
