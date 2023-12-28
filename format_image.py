from PIL import Image
from torchvision.transforms import ToTensor


class FormatImage:
    def __init__(self, image_size):
        self.to_tensor = ToTensor()
        self.image_size = image_size

    def __call__(self, image):
        width, height = image.size
        max_dimension = max(width, height)

        scale_factor = self.image_size / max_dimension

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        scaled_image = image.resize((new_width, new_height))

        new_image = Image.new("RGB", (self.image_size, self.image_size))
        padding = (
            (self.image_size - new_width) // 2,
            (self.image_size - new_height) // 2,
        )
        new_image.paste(scaled_image, padding)

        return self.to_tensor(new_image)
