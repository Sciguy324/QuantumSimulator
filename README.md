# QuantumSimulator

<img src="Icons/Icon.svg" title="" alt="" width="78">

This package contains tools for running a finite-difference method simulation of the Schrödinger equation.  Written for Advanced Quantum Mechanics in Spring 2023.

# Example Simulations

Examples for using the simulator can be found in the `Examples` folder.  Included are:
* 1D harmonic oscillator (eigenstate)
* 1D harmonic oscillator (superposition)
* 1D infinite potential well
* 2D infinite potential well
* 2D "approximate" point charge
* 2D "approximate" double slit

These example simulations include a window for viewing their evolution.  For instance, here are the results from the 2D potential well simulation:

<img src="Examples/example.gif" title="2D potential well" alt="Animated GIF of a 2D potential well in a superposition state." width="350">

# Requirements

This was written with the following python version and packages.  Others may also work, but have not been tested.

| Name   | Version |
| ------ | ------- |
| Python | 3.9.13  |
| Numpy  | 1.24.2  |
| Scipy  | 1.10.1  |
| Pyglet | 2.0.6   |
