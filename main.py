# ------------------------------------------------------------------------------
# Import required libraries
# ------------------------------------------------------------------------------

# FastAPI for web API creation
from fastapi import FastAPI  # The main web framework
from fastapi.responses import FileResponse  # Used to send file as API response

# Pydantic for validating request data
from pydantic import BaseModel  # Used to define and validate request body schema

# Matplotlib for drawing the neural network
import matplotlib.pyplot as plt

# Utilities for unique file names and file path handling
import uuid  # To generate unique IDs for each diagram
import os    # For file operations and temporary paths

# ------------------------------------------------------------------------------
# Step 1: Create FastAPI app instance
# ------------------------------------------------------------------------------
app = FastAPI(title="Tiny NN Diagram Generator API",
    description="Generate simple feedforward neural network SVG diagrams from layer sizes.",
    version="1.0.0")

# ------------------------------------------------------------------------------
# Step 2: Define request schema using Pydantic
# ------------------------------------------------------------------------------

class NeuralNetInput(BaseModel):
    layer_sizes: list[int]               # Required: Number of nodes in each layer (e.g., [4, 6, 3])
    colors: list[str] = ['red', 'blue', 'green', 'purple']  # Optional: colors for each layer
    bias_color: str = 'gray'             # Optional: color of the bias nodes

# ------------------------------------------------------------------------------
# Step 3: Drawing logic for neural network diagram
# ------------------------------------------------------------------------------

def draw_dynamic_neural_net(layer_sizes,
                            colors=['red', 'blue', 'green', 'purple'],
                            bias_color='gray',
                            save_path="neural_network_diagram.svg"):
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')  # Hide axes and labels
    ax.set_aspect('equal')  # Keep node circles round

    # Define spacing and appearance settings
    num_layers = len(layer_sizes)  # Total number of layers
    x_spacing = 3                  # Horizontal space between layers
    y_spacing = 1.5                # Vertical space between nodes in a layer
    node_radius = 0.3              # Size of each node circle
    line_width = 0.5               # Thickness of connection lines

    # Calculate x-positions for each layer
    x_positions = [i * x_spacing for i in range(num_layers)]

    # Dictionary to store node (x, y) positions
    node_positions = {}

    # Helper function to define label prefix
    def get_prefix(layer_idx):
        if layer_idx == 0:
            return 'x'  # Input layer: x1, x2, ...
        elif layer_idx == num_layers - 1:
            return 'o'  # Output layer: o1, o2, ...
        else:
            return 'h'  # Hidden layers: h1, h2, ...

    # Function to draw nodes in a layer
    def draw_layer(layer_idx, num_nodes, color):
        # Compute vertically centered y-positions
        y_positions = [-(i * y_spacing - (num_nodes - 1) * y_spacing / 2) for i in range(num_nodes)]

        for i, y in enumerate(y_positions):
            x = x_positions[layer_idx]
            node_id = f"L{layer_idx}_N{i}"  # e.g., "L1_N2"
            node_positions[node_id] = (x, y)  # Save (x, y) for line drawing

            # Draw the node (circle)
            ax.add_patch(plt.Circle((x, y), radius=node_radius, color=color, ec='black', lw=1.0, zorder=3))

            # Assign label: h1 on top, h2 below, etc.
            label = f"{get_prefix(layer_idx)}{i + 1}"
            ax.text(x, y, label, ha='center', va='center', color='white', fontsize=10, zorder=4)

    # Draw all layers
    for layer_idx, num_nodes in enumerate(layer_sizes):
        color = colors[layer_idx] if layer_idx < len(colors) else 'blue'  # Fallback color
        draw_layer(layer_idx, num_nodes, color)

    # Draw bias nodes (above each hidden/output layer)
    bias_nodes = {}
    max_y = max(layer_sizes) * y_spacing / 2  # Max vertical position
    for l in range(1, num_layers):  # Skip input layer
        x = (x_positions[l - 1] + x_positions[l]) / 2  # Between layers
        y = max_y + 0.8  # Slightly above the top node
        bias_id = f"b{l}"
        bias_nodes[bias_id] = (x, y)

        # Draw the bias node (circle and label)
        ax.add_patch(plt.Circle((x, y), radius=node_radius, color=bias_color, ec='black', lw=1.0, zorder=3))
        ax.text(x, y, bias_id, ha='center', va='center', color='white', fontsize=10, zorder=4)

    # Draw connections between layers
    for l in range(num_layers - 1):
        for i in range(layer_sizes[l]):
            for j in range(layer_sizes[l + 1]):
                src = node_positions[f"L{l}_N{i}"]
                dst = node_positions[f"L{l+1}_N{j}"]
                ax.plot([src[0], dst[0]], [src[1], dst[1]], color='black', linewidth=line_width, zorder=1)

        # Connect bias node to all next-layer nodes with dashed lines
        for j in range(layer_sizes[l + 1]):
            src = bias_nodes[f"b{l+1}"]
            dst = node_positions[f"L{l+1}_N{j}"]
            ax.plot([src[0], dst[0]], [src[1], dst[1]], color='black', linewidth=line_width, linestyle='--', zorder=1)

    # Set visible canvas area
    ax.set_xlim(-1, x_positions[-1] + 2)
    ax.set_ylim(-max_y - 1.5, max_y + 2)

    # Save diagram to SVG file
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()  # Free up memory
    return save_path  # Return the saved path

# ------------------------------------------------------------------------------
# Step 4: API endpoint to generate SVG and return it
# ------------------------------------------------------------------------------

@app.post("/generate-diagram")
async def generate_svg(input_data: NeuralNetInput):
    # Generate a unique file name using UUID
    file_name = f"neural_net_{uuid.uuid4().hex}.svg"

    # Save in /tmp directory (safe for cloud + local)
    save_path = os.path.join("/tmp", file_name)

    # Generate the diagram
    draw_dynamic_neural_net(
        layer_sizes=input_data.layer_sizes,
        colors=input_data.colors,
        bias_color=input_data.bias_color,
        save_path=save_path
    )

    # Send back the SVG file as a downloadable response
    return FileResponse(save_path, media_type="image/svg+xml", filename=file_name)
# ------------------------------------------------------------------------------
# Step 5: Root GET endpoint for health check and usage info
# ------------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "message": "Tiny NN Diagram Generator API is running!",
        "usage": {
            "POST /generate-diagram": {
                "description": "Generates a neural network SVG",
                "example_input": {
                    "layer_sizes": [4, 6, 3],
                    "colors": ["red", "blue", "green"],
                    "bias_color": "gray"
                }
            },
            "docs": "/docs"
        }
    }
