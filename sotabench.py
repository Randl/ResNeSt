from torchbench.image_classification import ImageNet
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import PIL
import urllib.request
import torch
import torchvision.models as models


class ECenterCrop:
    """Crop the given PIL Image and resize it to desired size.
    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
            it is used for both directions
    Returns:
        PIL Image: Cropped image.
    """
    def __init__(self, imgsize):
        self.imgsize = imgsize
        self.resize_method = transforms.Resize((imgsize, imgsize), interpolation=PIL.Image.BICUBIC)

    def __call__(self, img):
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        img = img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
        return self.resize_method(img)


# Model 1
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
        ECenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
model = torch.hub.load('zhanghang1989/ResNeSt',  'resnest50', pretrained=True)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ResNeSt-50',
    paper_arxiv_id='2004.08955',
    input_transform=input_transform,
    batch_size=32,
    num_gpu=1,
    model_description="Official weights from the author's of the paper.",
)
torch.cuda.empty_cache()


# Model 2
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
        ECenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
model = torch.hub.load('zhanghang1989/ResNeSt',  'resnest101', pretrained=True)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ResNeSt-101',
    paper_arxiv_id='2004.08955',
    input_transform=input_transform,
    batch_size=32,
    num_gpu=1,
    model_description="Official weights from the author's of the paper."
)
torch.cuda.empty_cache()


# Model 1
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
        ECenterCrop(320),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
model = torch.hub.load('zhanghang1989/ResNeSt',  'resnest200', pretrained=True)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ResNeSt-200',
    paper_arxiv_id='2004.08955',
    input_transform=input_transform,
    batch_size=32,
    num_gpu=1,
    model_description="Official weights from the author's of the paper."
)
torch.cuda.empty_cache()


# Model 1
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
        ECenterCrop(416),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
model = torch.hub.load('zhanghang1989/ResNeSt',  'resnest269', pretrained=True)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ResNeSt-269',
    paper_arxiv_id='2004.08955',
    input_transform=input_transform,
    batch_size=32,
    num_gpu=1,
    model_description="Official weights from the author's of the paper."
)
torch.cuda.empty_cache()