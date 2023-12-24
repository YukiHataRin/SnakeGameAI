from PIL import Image, ImageDraw, ImageFont

# Function to draw a fully connected neural network layer
def draw_layer(draw, origin, neurons, next_neurons, layer_name, weights_label, bias_label, ellipse_radius=20):
    # Draw the neurons
    x, y = origin
    y_step = 100  # Vertical space between neurons
    for i in range(neurons):
        if i == neurons - 1:  # Last neuron, draw ellipsis to imply continuation
            draw.text((x - 5, y + (i * y_step) - 10), "...", fill="black")
        else:
            draw.ellipse((x - ellipse_radius, y + (i * y_step) - ellipse_radius,
                          x + ellipse_radius, y + (i * y_step) + ellipse_radius), outline="black", width=2)
            # Connect to the next layer if not the output layer
            if next_neurons is not None:
                next_x, next_y = x + 300, origin[1]  # Horizontal space between layers
                for j in range(next_neurons):
                    next_y_step = 100 if next_neurons != 3 else 150  # Bigger step for last layer for visualization purposes
                    draw.line((x + ellipse_radius, y + (i * y_step),
                               next_x - ellipse_radius, next_y + (j * next_y_step)), fill="black")

    # Labels for layer name, weights, and bias
    draw.text((x - 20, y - 75), layer_name, fill="black")
    draw.text((x + 100, y + (neurons * y_step / 2) - 50), weights_label, fill="black")
    draw.text((x - 50, y + (neurons * y_step) + 20), bias_label, fill="black")


# Create an image with white background
width, height = 1000, 1000
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

# Define the structure of the FCNN
layers = [
    {"neurons": 5, "next_neurons": 4, "name": "input\n(256)", "weights": "(256x11)", "bias": "(256)"},
    {"neurons": 4, "next_neurons": 4, "name": "linear1", "weights": "(128x256)", "bias": "(128)"},
    {"neurons": 4, "next_neurons": 3, "name": "linear2", "weights": "(64x128)", "bias": "(64)"},
    {"neurons": 3, "next_neurons": None, "name": "linear3", "weights": "(3x64)", "bias": "(3)"},
]

# Draw the layers
origin_x, origin_y = 100, 150
for i, layer in enumerate(layers):
    draw_layer(draw, (origin_x + (i * 300), origin_y), layer["neurons"], layer["next_neurons"],
               layer["name"], layer["weights"], layer["bias"])
    
image.save('test.png')