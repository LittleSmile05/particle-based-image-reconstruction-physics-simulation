# Particle-Based Image Reconstruction Physics Simulation

A Python simulation that uses particles (balls) and physics principles to reconstruct images. Each particle interacts with gravity, collisions, and spring forces to gradually form the original image.

---

## Features

- **Physics-Based Motion**: Uses Verlet integration for realistic particle movement.
- **Adaptive Particle Count**: Adjusts the number of particles based on image complexity.
- **Collision Handling**: Particles bounce off each other and screen boundaries.
- **Spring Forces**: Particles are pulled toward target positions for image formation.
- **Image Sampling**: Particles are spawned according to image importance (edges, color, saturation).
- **CSV Export/Import**: Save and reload particle positions for repeated simulations.
- **Customizable Parameters**: Gravity, spring strength, particle size, and simulation steps.

---

## Installation & Usage

```bash
# Install dependencies
pip install pygame pillow numpy opencv-python

# Run the simulation with default settings
python main.py --image input_image.jpg --mode SPAWN

# Optional command-line arguments:
# --image: Path to input image (default: input_image.jpg)
# --mode: Simulation mode (SPAWN to create particles, LOAD to load from CSV)
# --steps: Number of simulation steps (default: 400)
# --min-particles: Minimum particle count
# --max-particles: Maximum particle count
# --spring-strength: Spring force strength
# --gravity: Gravity strength
# --preprocess: Preprocess image for better edge and color handling

# Controls:
# S      -> Save current particle positions to CSV
# R      -> Reset particles to target positions
# SPACE  -> Pause/Resume simulation
# Q/ESC  -> Quit


<img width="1919" height="973" alt="Screenshot 2025-10-16 192705" src="https://github.com/user-attachments/assets/90267d4a-05d8-45de-b621-2150ca5be1c6" />
