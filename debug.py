import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
from transformers import CLIPModel, CLIPConfig, CLIPTokenizer, \
    CLIPProcessor, CLIPFeatureExtractor


class CLIPLoss(torch.nn.Module):
    def __init__(self, cache_dir):
        super(CLIPLoss, self).__init__()
        self.model, self.processor, self.tokenizer = get_clip(cache_dir)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, image, text):
        image = torch.nn.functional.upsample_bilinear(image, (224, 224))
        # similarity = 1 - self.model(image, text)[0] / 100

        inputs = self.processor(text, image, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image / 100
        return 1 - logits_per_image

def get_clip(cache_dir, model_name="openai/clip-vit-base-patch32"):
    config = CLIPConfig.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir, config=config)
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir, config=config)
    processor.feature_extractor = CLIPFeatureExtractor(processor.feature_extractor)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, cache_dir=cache_dir, config=config)

    return model, processor, tokenizer


def tensor_to_images(tensor):
    images = []
    for i in range(tensor.shape[0]):
        image = transforms.ToPILImage()(tensor[i].data).convert('RGB')
        images.append(image)
    return images


class StyleScorer(nn.Module):
    def __init__(self, cache_dir, device='cuda:0'):
        super().__init__()
        self.model, self.processor, self.tokenizer = get_clip(cache_dir)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(device)

    def forward(self, images, text):
        inputs = self.processor(text, images, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        return 1 - logits_per_image.softmax(dim=1)


image_pil = Image.open('data/sample.png').convert('RGB')
prefix = 'a model of a'
suffix = 'on a white background'
text = ['dog', 'llama', 'car', 'red car', 'green truck', 'vehicle']
text = [prefix + ' ' + t + ' ' + suffix for t in text]
image = transforms.ToTensor()(image_pil).unsqueeze(0)
# plt.imshow(image[0].permute(1, 2, 0).detach().cpu())
# plt.show()

cache_dir = '../clip_models'
model, processor, tokenizer = get_clip(cache_dir)

# image = torch.nn.functional.upsample_bilinear(image, (224, 224))
inputs = processor(text, image_pil, return_tensors="pt", padding=True).to(model.device)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image / 100
loss1 = 1 - logits_per_image

s = StyleScorer(cache_dir).cuda()
inputs = s.processor(text, image_pil, return_tensors="pt", padding=True).to(s.model.device)
outputs = s.model(**inputs)
logits_per_image = outputs.logits_per_image
loss2 = 1 - logits_per_image.softmax(dim=1)
print('hi')