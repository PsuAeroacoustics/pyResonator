# Resonator: Acoustic Resonator and Sound Absorption Modeling

A Python library for modeling and analyzing acoustic resonators, perforated facesheets, and porous bulk materials for sound absorption applications. This project provides computational tools to predict complex acoustic impedance and absorption coefficients for various acoustic treatment designs.

## Overview

This project implements sophisticated acoustic modeling techniques for designing and analyzing acoustic materials and structures. It includes implementations of:

- **Helmholtz Resonators**: Cavity-backed neck systems for frequency-selective sound absorption
- **Perforated Facesheets**: Multi-parameter models for perforated panel impedance
- **Porous Bulk Materials**: Biot-Allard model for fibrous/foam absorption materials
- **Composite Systems**: Combined impedance analysis for multi-layer acoustic treatments

The library supports multiple theoretical models with varying levels of complexity and accuracy, from basic acoustic element approximations to rigorous waveguide solutions with thermoviscous losses.

## Features

### Acoustic Modeling Capabilities

- **Multiple Resonator Models**:
  - Waveguide (WG) model - exact solution for plane waves in cylindrical tubes
  - Basic Acoustic Element (BAE) model - simplified low-frequency approximation
  - Kirchoff model - with thermal and viscous losses via Bessel functions

- **Facesheet Models**:
  - 2-Parameter model for perforated facesheets
  - Atalla and Sgard (AS) semi-empirical model
  - Sound pressure level (SPL) and Mach number dependency

- **Porous Material Models**:
  - Delany-Bazley model - empirical method
  - Biot-Allard model - rigorous 4-parameter theory
  - Automatic parameter tuning via least-squares optimization

### Advanced Features

- Spherical radiation impedance corrections
- Interior correction for closely-spaced resonators
- Thermoviscous boundary layer effects
- Lookup tables for propagation constants (Tijdeman)
- Normal incidence absorption coefficient calculation
- Multi-resonator array analysis

## Installation

### Requirements

The following dependencies are required: 

- numpy
- matplotlib
- scipy

These can be installed as follows
```bash
pip install -r requirements.txt
```

## Project Structure

```
resonator/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── resonator.py                        # Main resonator and facesheet classes
├── example.py                          # Example usage and basic demonstrations
├── facesheet_validation.py             # Facesheet model validation
├── porous_bulk_mat_validation.py       # Porous material model validation
├── variable_depth_resonator_validation.py  # Variable depth resonator designs
├── Tijdeman_gamma/                     # Lookup tables for propagation constants
│   ├── re_gamma.txt                    # Real part of propagation constant
│   └── imag_gamma.txt                  # Imaginary part of propagation constant
└── validation_data/                    # Experimental validation datasets
    ├── m1_*.csv                        # Material 1 impedance data
    ├── mfoam_*.csv                     # Foam material data
    └── qf130_*.csv                     # QF-130 fiberglass data
```

## Usage Guide

### Basic Helmholtz Resonator Example

```python
import numpy as np
import resonator as res

# Define frequency range
f = np.arange(100, 5000, 10)

# Create resonator with:
# - neck radius: 1 cm
# - neck length: 5 cm
# - cavity radius: 1 cm (same as neck)
# - cavity length: 10 cm
helm = res.resonator(
    a_n=0.01,      # neck radius [m]
    L_n=0.05,      # neck length [m]
    a_c=0.01,      # cavity radius [m]
    L_c=0.10       # cavity length [m]
)

# Calculate impedance using Kirchoff model with losses
helm.set_Z(f, model='Kirchoff', loss=True, rad=True)

# Get results
impedance = helm.get_Z()
absorption = helm.get_alpha()

# Plot results
helm.plot(xlim=[100, 5000])
```

### Perforated Facesheet Example

```python
# Create facesheet with:
# - thickness: 1 mm
# - perforation radius: 0.5 mm
# - porosity (open area ratio): 7%
facesheet = res.fs(
    t=1e-3,        # thickness [m]
    r=0.5e-3,      # perforation radius [m]
    phi=0.07       # porosity
)

# Set impedance using 2-Parameter model
facesheet.set_Z(f, model='2P', SPL=0, M=0, Z_cav=0)

# Access impedance and absorption
Z_fs = facesheet.get_Z()
alpha_fs = facesheet.get_alpha()
```

### Combined Resonator + Facesheet System

```python
# Create resonator and facesheet
helm = res.resonator(a_n=0.01, L_n=0.05, a_c=0.01, L_c=0.10)
fs = res.fs(t=1e-3, r=0.5e-3, phi=0.07)

# Set impedances
helm.set_Z(f, model='Kirchoff', loss=True)
fs.set_Z(f, model='2P', Z_cav=helm.Z)  # Facesheet sees resonator impedance

# Combined impedance
Z_combined = helm.Z + fs.Z
alpha_combined = 1 - abs((Z_combined - 1) / (Z_combined + 1))**2
```

### Porous Material Tuning

```python
# Create resonator with porous material properties
porous = res.resonator(
    t=0.05,                # thickness [m]
    phi=0.95,              # porosity
    sigma=5000,            # flow resistivity [MKS Rayls/m]
    q=1.3,                 # tortuosity
    s_b=1.37               # shape factor
)

# Tune parameters to match experimental data
porous.tune_params(
    f=f,
    val_data_Re=experimental_resistance,
    val_data_Im=experimental_reactance,
    bnds=([1e3, 0.5, 0.5, 0.2], [1e5, 3, 3, 1])
)

# Calculate impedance
porous.set_Z(f)
```

### Variable Depth Resonator Array

```python
# Create array of different-depth resonators
resonator_depths = np.array([3.751, 2.701, 2.110, 1.731, 1.350]) / 39.37  # in meters
cavity_radius = 0.00381  # 3/16 inch

# Initialize all resonators
resonators = []
for depth in resonator_depths:
    helm = res.resonator(
        a_n=cavity_radius,
        L_n=depth/2,
        a_c=cavity_radius,
        L_c=depth/2
    )
    helm.set_Z(f, model='Kirchoff')
    resonators.append(helm)

# Combine impedances (parallel combination)
N_total = 25
N_per_depth = N_total / len(resonator_depths)
Z_array = sum(
    N_per_depth * helm.Z**-1 
    for helm in resonators
)**-1
alpha_array = 1 - abs((Z_array - 1) / (Z_array + 1))**2
```

## Class Reference

### `resonator` Class

Main class for modeling Helmholtz resonators and porous materials.

#### Initialization Parameters

```python
resonator(
    c=340,              # Speed of sound [m/s]
    P=101325,           # Atmospheric pressure [Pa]
    gamma=1.4,          # Specific heat ratio
    rho=1.125,          # Density [kg/m³]
    nu=14.88e-6,        # Kinematic viscosity [m²/s]
    Pr=0.71,            # Prandtl number
    
    # Helmholtz resonator parameters:
    a_n=None,           # Neck radius [m]
    L_n=None,           # Neck length [m]
    a_c=None,           # Cavity radius [m]
    L_c=None,           # Cavity length [m]
    
    # Porous material parameters:
    t=0.01,             # Thickness [m]
    sigma=None,         # Flow resistivity [MKS Rayls/m]
    phi=None,           # Porosity [0-1]
    q=1.30384,          # Tortuosity
    s_b=1.37            # Shape factor
)
```

#### Key Methods

- **`set_Z(f, model='Kirchoff', rad=False, loss=False, interior=False, table=False)`**
  - Calculate complex impedance
  - `model`: 'WG' (waveguide), 'BAE' (basic element), 'Kirchoff' (with losses)
  - `rad`: Include radiation impedance
  - `loss`: Include thermoviscous losses
  - `interior`: Correction for interior resonators
  - `table`: Use Tijdeman lookup table for propagation constants

- **`set_alpha()`**
  - Calculate normal incidence absorption coefficient

- **`get_Z()`**
  - Returns computed complex impedance

- **`get_alpha()`**
  - Returns computed absorption coefficient

- **`tune_params(f, val_data_Re, val_data_Im, bnds)`**
  - Optimize porous material parameters to match experimental data
  - Uses least-squares optimization

- **`plot(xlim=[100, 10e3])`**
  - Plot impedance (real/imaginary) and absorption coefficient

- **`set_vdata(vdat_file)`**
  - Load validation data from file

- **`get_gamma_tab()`** (static)
  - Load Tijdeman propagation constant lookup tables

### `fs` Class

Perforated facesheet model.

#### Initialization Parameters

```python
fs(
    t,              # Facesheet thickness [m]
    r,              # Perforation radius [m]
    phi,            # Porosity (open area ratio)
    c=340,          # Speed of sound [m/s]
    rho=1.125,      # Density [kg/m³]
    nu=14.88e-6     # Kinematic viscosity [m²/s]
)
```

#### Key Methods

- **`set_Z(f, model='2P', SPL=0, M=0, Z_cav=0)`**
  - Calculate facesheet impedance
  - `model`: '2P' (2-parameter) or 'AS' (Atalla-Sgard)
  - `SPL`: Sound pressure level [dB]
  - `M`: Mach number (for aerodynamic effects)
  - `Z_cav`: Impedance behind facesheet (cavity/resonator)

- **`get_Z()`** / **`get_alpha()`**
  - Return impedance and absorption coefficient

## Example Scripts

### `example.py`
Demonstrates:
- Basic resonator impedance calculation
- Uniform and variable-depth resonator arrays
- Comparison of WG vs. BAE models
- Absorption coefficient plots

### `facesheet_validation.py`
Validates facesheet model against theoretical predictions

### `porous_bulk_mat_validation.py`
Tests porous material models with experimental data

### `variable_depth_resonator_validation.py`
Designs and validates variable-depth resonator arrays
