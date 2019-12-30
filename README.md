# projects

## What is this?

This repository contains a collection of small side-projects I have written.
These mini-projects are curated and maintained by myself, but not all are under active development.
None of these are meant to be long-term commitments, but rather small exercises to keep my skills sharp and/or learn a new language, framework, algorithm or technique, or just to solve a problem.

## Why only one repository?

The mono-repository structure allows for projects to be collected in one place, so they can be much more easily mantained over time.

This many projects can really clutter a github page, and make it difficult to discern major from minor projects.
Here, I can maintain an overview of my past minor projects, and dedicate an entire repository to major projects under active development/maintenance.

Further, I do not have to deal with the major turnoffs for monorepos:
There's no issue of scalability, because any huge project can be split off into its own repo.
On that same note, any project requiring CI/CD or collaboration from peers should also be its own repo.

At the end of the day, this choice was made because it provides more benefits than drawbacks _for this use case_.

## Projects

- C
  - [Triangle Image Evolution](C/Triangle-Image-Evolution/) (`cairo`)
  - [Langton's Ant](C/Langtons-Ant/)
- C++
  - [Conway's Game of Life implementation](C++/Game-of-Life/) (`SFML`)
  - [NES Tetris](C++/Tetris/) (`SFML`)
- Julia
  - [OIST Skill-Pill Notes](Julia/Skill-Pill/)
  - [OIST Scientific Computing in Julia Workshop](Julia/Workshop/)
  - [Julia Set Renderer](Julia/Fractal-Render/) (`CSFML.jl`)
  - [Render Engine](Julia/Graphics/)
- Python
  - [Split-Step Fourier Method NLSE solver](Python3/Split-Operator-Solver/)
  - [Video to Ascii converter](Python3/video2ascii/) (`curses`)
  - [ResNet-18 Implementation](Python3/ResNet-18/) (`PyTorch`)
  - [Image Compression with Convolutional Autoencoders](Python3/Conv-Autoencoder/) (`PyTorch`)
  - [UTSC Available Room Finder](Python3/UTSC-Room-Finder/) (`requests`)
  - [Noise Algorithms](Python3/Noise/) (`numpy`, `matplotlib`)

