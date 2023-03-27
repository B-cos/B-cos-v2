# B-cos Networks v2
M. Böhle, N. Singh, M. Fritz, B. Schiele.

Improved B-cos Networks.

If you want to take a quick look at the explanations the models generate, 
you can try out the Gradio web demo on [🤗 Spaces](https://huggingface.co/spaces/nps1ngh/B-cos).

If you prefer a more hands-on approach, 
you can take a look at the [demo notebook on Colab](https://colab.research.google.com/drive/1bdc1zdIVvv7XUJj8B8Toe6VMPYAsIT9w?usp=sharing).
or load the models directly via torch hub as explained below.


## Quick Start
Loading the models via torch hub is as easy as:

```python
import torch

# list all available models
torch.hub.list('B-cos/B-cos-v2')

# load a pretrained model
model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
```

Inference and explanation visualization is as simple as:
```python
from PIL import Image
import matplotlib.pyplot as plt

# load image
img = model.transform(Image.open('cat.jpg'))
img = img[None].requires_grad_()

# predict and explain
model.eval()
expl_out = model.explain(img)
print("Prediction:", expl_out["prediction"])  # predicted class idx
plt.imshow(expl_out["explanation"])
plt.show()
```

See the [demo notebook](https://colab.research.google.com/drive/1bdc1zdIVvv7XUJj8B8Toe6VMPYAsIT9w?usp=sharing) for more details.



# *Stay tuned, more info to come soon!*