from PIL import Image, ImageDraw
from bounding_box import BoundingBox

# Wrote based on this tutorial: https://www.blog.pythonlibrary.org/2021/02/23/drawing-shapes-on-images-with-python-and-pillow/
def draw(analyze_results, image_file_path, output_file_path):
    image = Image.open(image_file_path)
    draw = ImageDraw.Draw(image)
    color = 'red'

    for page_result in analyze_results.read_results:
        for line in page_result.lines:
            boundaries = line.bounding_box
            bounding_box = BoundingBox(boundaries)
            # It takes X0, Y0 (the starting point) and X1, Y1 (the ending point). Which in my opinion is weird.
            draw.rectangle((bounding_box.x,bounding_box.y,bounding_box.x + bounding_box.width, bounding_box.y + bounding_box.height), outline=color, width=2)
    image.save(output_file_path)