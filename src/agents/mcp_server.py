"""
Pillow based image creation using MCP server
"""

from mcp.server.fastmcp import FastMCP
from PIL import Image, ImageDraw, ImageFont

mcp = FastMCP("image-creator")

@mcp.tool()
def create_canvas(width: int, height: int, output_path: str = 'output.png') -> str:
    """Create a canvas with the specified width and height"""
    image = Image.new("RGB", (width, height), "white")
    image.save(output_path)
    return f"Created {width}x{height} white canvas at '{output_path}'"


@mcp.tool()
def get_image_info(image_path: str = 'output.png') -> str:
    """Get information about an existing image"""
    try:
        image = Image.open(image_path)
        return f"Image '{image_path}': {image.width}x{image.height}, mode={image.mode}"
    except FileNotFoundError:
        return f"Error: Image '{image_path}' not found"

@mcp.tool()
def add_rectangle(x: int, y: int, width: int, height: int, color: str = "red", image_path: str = 'output.png') -> str:
    """Add a rectangle to the image"""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    draw.rectangle([x, y, x + width, y + height], fill=color)
    image.save(image_path)
    return f"Added {color} rectangle at ({x},{y}) with size {width}x{height} to '{image_path}'"

@mcp.tool()
def write_text(text: str, x: int, y: int, font_size: int = 48, color: str = "black", image_path: str = 'output.png') -> str:
    """Write text on the image"""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text([x, y], text, fill=color, font=font)
    image.save(image_path)
    return f"Added text '{text}' in {color} at position ({x},{y}) with font size {font_size} to '{image_path}'"

if __name__ == "__main__":
    mcp.run(transport="stdio")