import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import lambertw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def W(x):
    """Vectorized Lambert W function"""
    return lambertw(x).real

def dfdx(x, c1):
    xi = np.sqrt(2 * (np.log(x + 1) - c1 + 1))
    return 2 / (W(np.exp(xi**2/2 + c1 - 1)) + 1) - 1

class FluidFlow(torch.nn.Module):
    def __init__(self, grid_size, c1):
        super(FluidFlow, self).__init__()
        self.grid_size = grid_size
        self.c1 = c1
        self.grid = torch.zeros((grid_size, grid_size), device=device)
   
    def initialize_grid(self):
        """Initialize the grid with a more complex condition"""
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, y)
       
        # Create a more complex initial condition
        initial_condition = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        initial_condition += 0.5 * np.sin(4 * np.pi * X) * np.cos(4 * np.pi * Y)
       
        # Normalize to [0, 1] range
        initial_condition = (initial_condition - initial_condition.min()) / (initial_condition.max() - initial_condition.min())
       
        self.grid = torch.tensor(initial_condition, device=device)
       
        # Set boundaries to 0
        self.grid[0, :] = self.grid[-1, :] = self.grid[:, 0] = self.grid[:, -1] = 0.0
   
    def step(self, dt):
        """Perform one step of the simulation"""
        grid_np = self.grid.cpu().numpy()
       
        # Calculate the change in x
        dx = dfdx(grid_np, self.c1) * dt
       
        # Update x
        grid_np += dx
       
        # Apply boundary conditions
        grid_np[0, :] = grid_np[-1, :] = grid_np[:, 0] = grid_np[:, -1] = 0.0
        self.grid = torch.tensor(grid_np, device=device)
   
    def run_simulation(self, steps, dt):
        """Run the simulation for a given number of steps"""
        for _ in range(steps):
            self.step(dt)
            yield self.grid.cpu().numpy()

# Set up simulation parameters
grid_size = 100
c1 = 0.5
steps = 200
dt = 0.01

# Create the simulation
simulation = FluidFlow(grid_size, c1)
simulation.initialize_grid()

# Set up the figure and axis for animation
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(simulation.grid.cpu().numpy(), cmap='viridis', animated=True)
fig.colorbar(im)

# Animation update function
def update(frame):
    im.set_array(frame)
    return [im]

# Create the animation
anim = FuncAnimation(fig, update, frames=simulation.run_simulation(steps, dt),
                     interval=50, blit=True)

plt.title('Fluid Flow Simulation')
plt.close()  # Prevents duplicate display in Jupyter notebooks

# Save the animation as a gif
anim.save('fluid_flow_animation.gif', writer='pillow', fps=20)

print("Animation saved as 'fluid_flow_animation.gif'")
