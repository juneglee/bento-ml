import io
import bentoml
import numpy as np
from bentoml.io import Image, NumpyNdarray
from PIL.Image import Image as PILImage

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

runner = bentoml.pytorch.get("resnet18_v1").to_runner()

svc = bentoml.Service(
    name="resnet",
    runners=[
        runner,
    ],
)

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])

class_names = {
    "0": "ant", "1": "bee",
}

class image_datast(Dataset):
    def __init__(self, image, transform =None):
        self.image = image
        self.transform = transform

    def __len__(self):
        buffer = io.BytesIO()
        self.image.save(buffer, format='JPEG', quality=75)
        image_object = buffer.getbuffer()
        return len(image_object)

    def __getitem__(self, idx):
        img = self.image
        image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image

@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
async def predict_image(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    image_lo = image_datast(f, transform= transform)
    in_loader = DataLoader(image_lo, batch_size=1, shuffle=False, num_workers=0)
    image = next(iter(in_loader))

    output_tensor = await runner.async_run(image)
    _, pred = torch.max(output_tensor, 1)

    return np.array(class_names[str(pred.tolist()[0])])