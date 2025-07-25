# Import FastAPI framework and supporting utilities
from fastapi import FastAPI  # The main web framework
from fastapi.responses import FileResponse  # Used to send back a file as a response

# Import Pydantic for request validation
from pydantic import BaseModel  # Used to define and validate request body schema

# Import Matplotlib for drawing the neural network diagram
import matplotlib.pyplot as plt

# Utilities for generating unique filenames and working with file paths
import uuid  # For generating unique IDs
import os    # For file path handling

# ------------------------------------------------------------------------------
# Step 1: Create the FastAPI app instance
# ------------------------------------------------------------------------------
app = FastAPI()

# ------------------------------------------------------------------------------
# Step 2: Define the request body schema using Pydantic
# ------------------------------------------------------------------------------
class NeuralNetInput(BaseModel):
    layer_sizes: list[int]               # Required: List of number of nodes in each layer (e.g., [4, 8, 3])
    colors: list[str] = ['red', 'blue', 'blue', 'green']  # Optional: list of colors for each layer
    bias_color: str = 'gray'             # Optional: color for the bias nodes

# ------------------------------------------------------------------------------
# Step 3: Function to draw the neural network diagram and save as an SVG
# ------------------------------------------------------------------------------
def draw_dynamic_neural_net(layer_sizes,
                            colors=['red', 'blue', 'blue', 'green'],
                            bias_color='gray',
                            save_path="neural_network_diagram.svg"):
    # Create a new figure and axes for drawing
    fig, ax = plt.subplots(figsize=(10, 8))  # Set figure size in inches
    ax.axis('off')  # Turn off axes (no x/y labels)
    ax.set_aspect('equal')  # Ensure circles look round (not stretched)

    # Define constants for layout
    num_layers = len(layer_sizes)  # Total number of layers
    x_spacing = 3                  # Horizontal space between layers
    y_spacing = 1.5                # Vertical space between nodes in a layer
    node_radius = 0.3              # Radius of each node (circle)
    line_width = 0.5               # Line thickness for edges

    # X-coordinates for each layer
    x_positions = [i * x_spacing for i in range(num_layers)]

    # Dictionary to store node positions for later line drawing
    node_positions = {}

    # Helper function to return label prefix based on layer type
    def get_prefix(layer_idx):
        if layer_idx == 0:
            return 'x'  # input layer
        elif layer_idx == num_layers - 1:
            return 'o'  # output layer
        else:
            return f'h{layer_idx}'  # hidden layers

    # Inner function to draw all nodes in a single layer
    def draw_layer(layer_idx, num_nodes, color):
        # Compute y-positions so the layer is vertically centered
        y_positions = [-(i * y_spacing - (num_nodes - 1) * y_spacing / 2) for i in range(num_nodes)]
        for i, y in enumerate(y_positions):
            x = x_positions[layer_idx]
            node_id = f"L{layer_idx}_N{i}"  # Unique ID like "L1_N3"
            node_positions[node_id] = (x, y)  # Save the (x, y) position
            # Draw the node (circle)
            ax.add_patch(plt.Circle((x, y), radius=node_radius, color=color, ec='black', lw=1.0, zorder=3))
            # Add label like x1, h1, o1 inside the node
            label = f"{get_prefix(layer_idx)}{i+1}"
            ax.text(x, y, label, ha='center', va='center', color='white', fontsize=10, zorder=4)

    # Draw all layers using the draw_layer function
    for layer_idx, num_nodes in enumerate(layer_sizes):
        color = colors[layer_idx] if layer_idx < len(colors) else 'blue'  # Use fallback color if not enough given
        draw_layer(layer_idx, num_nodes, color)

    # Draw bias nodes above each layer (except input layer)
    bias_nodes = {}
    max_y = max(layer_sizes) * y_spacing / 2  # Top y-margin for placing bias nodes
    for l in range(1, num_layers):
        x = (x_positions[l - 1] + x_positions[l]) / 2  # Place bias between layers
        y = max_y + 0.8  # Slightly above the top node
        bias_id = f"b{l}"
        bias_nodes[bias_id] = (x, y)
        # Draw the bias node
        ax.add_patch(plt.Circle((x, y), radius=node_radius, color=bias_color, ec='black', lw=1.0, zorder=3))
        ax.text(x, y, bias_id, ha='center', va='center', color='white', fontsize=10, zorder=4)

    # Draw connections between nodes
    for l in range(num_layers - 1):  # for each pair of layers
        # Draw connections between each node in layer l and layer l+1
        for i in range(layer_sizes[l]):
            for j in range(layer_sizes[l + 1]):
                src = node_positions[f"L{l}_N{i}"]
                dst = node_positions[f"L{l+1}_N{j}"]
                ax.plot([src[0], dst[0]], [src[1], dst[1]], color='black', linewidth=line_width, zorder=1)

        # Connect bias node to all nodes in next layer (dashed line)
        for j in range(layer_sizes[l + 1]):
            src = bias_nodes[f"b{l+1}"]
            dst = node_positions[f"L{l+1}_N{j}"]
            ax.plot([src[0], dst[0]], [src[1], dst[1]], color='black', linewidth=line_width, linestyle='--', zorder=1)

    # Set visible area of the canvas
    ax.set_xlim(-1, x_positions[-1] + 2)
    ax.set_ylim(-max_y - 1.5, max_y + 2)

    # Save the diagram to an SVG file
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()  # Close the figure to release memory
    return save_path  # Return the path to the saved file

# ------------------------------------------------------------------------------
# Step 4: FastAPI endpoint to handle requests and return the generated SVG
# ------------------------------------------------------------------------------
@app.post("/generate-diagram")
async def generate_svg(input_data: NeuralNetInput):
    # Generate a unique filename using UUID to avoid overwriting
    file_name = f"neural_net_{uuid.uuid4().hex}.svg"
    save_path = os.path.join("/tmp", file_name)  # Use temp directory to store it

    # Call the function to draw the diagram and save the SVG
    draw_dynamic_neural_net(
        layer_sizes=input_data.layer_sizes,
        colors=input_data.colors,
        bias_color=input_data.bias_color,
        save_path=save_path
    )

    # Send the file back to the client as an HTTP response with the correct MIME type
    return FileResponse(save_path, media_type="image/svg+xml", filename=file_name)