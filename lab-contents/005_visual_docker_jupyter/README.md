The only prerequisite for this lab is Docker running. You do NOT need to have installed Python or anything like this in your dev environment.

# Build a local docker image and run a container

```bash
cat <<'EOF' | tee Dockerfile
FROM arm64v8/python:3.11.11
RUN pip install torchvision==0.21.0
RUN pip install lightning==2.5.0
RUN pip install lightning[extra]
RUN pip install jupyterlab==4.3.6
EOF
docker build --progress=plain --no-cache . -t torch-vision

docker run -it --rm -v $PWD:/usr/src -p 8888:8888 torch-vision /bin/bash -c "jupyter lab --notebook-dir=/usr/src --ip='*' --allow-root"
```

In the output, find the Jupyter Lab page (for example, http://localhost:8888/lab?token=497a91de59e60f9441d6a6c22125155d4004e627ef5d199a) and open it in a browser.

![Jupyter Lab Homepage](jupyter-lab-home.png)

In the Notebook section click on the Python 3 tile. You should see something like this:

![Empty Jupyter Notebook](jupyter-empty-notebook.png)

Run further Python code in the Jupyter lab.

# Prepare data for training

First, configure the session to show entire output from the Jupyter cells, instead of showing only output of the last line:

```py
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

Copy paste the code above to the first notebook cell and then click `Shift` + `Enter` for executing it:

![First Cell in Jupyter Lab](jupyter-first-cell.png)

Next, you can run the first piece of the code that loads and shows some examples from the training datasets and also shows images after pre-processing:

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Step 1: Dataset Exploration

# load dataset
sample_data = datasets.CIFAR10(root="data", train=True, download=True)
# show image classes
sample_data.classes
# first image
sample_data[0][0].resize([300, 300])
# first image class
sample_data.classes[sample_data[0][1]]
# second image
sample_data[1][0].resize([300, 300])
# second image class
sample_data.classes[sample_data[1][1]]
# third image
sample_data[2][0].resize([300, 300])
# third image class
sample_data.classes[sample_data[2][1]]


# Step 2: Dataset Preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])
# Download and load CIFAR-10 dataset
train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
# Show some images
to_img = ToPILImage()
#first image RGB tensor after transformation
train_data[0][0]
#first image after transformation
to_img(train_data[0][0]).resize([300, 300])
#second image after transformation
to_img(train_data[1][0]).resize([300, 300])
#third image after transformation
to_img(train_data[2][0]).resize([300, 300])
```

You should be able to see some samples of the training pictures before and after preprocessing:

![Sample Training Pictures](sample-training-pictures.png)

In this lab we are using CIFAR 10 training set with 50000 training 32x32 images, each associated with one of ten classes.

# Train the model

Next step is to do the training:

```py
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=7)

# Step 3: Define the Image Classification Model
class ImageClassifier(pl.LightningModule):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.nll_loss(outputs, labels)
        self.log('train_loss', loss)
        return loss

    # Validation step 
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = F.nll_loss(outputs, labels)
        self.log('val_loss', val_loss)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        test_loss = F.cross_entropy(outputs, labels)
        self.log('test_loss', test_loss, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

# Step 4: Training and Validation

val_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_train)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=7)

# Initialize the model
model = ImageClassifier()

# Create a Trainer object
trainer = Trainer(max_epochs=1, devices=1, accelerator="auto")
# Train the model
trainer.fit(model, train_loader, val_loader)
```

# Try it out

Now when the model got some training, we can test it out!

```py
# Looking at the non-seen data (train=False)
sample_data = datasets.CIFAR10(root="data", train=False, download=True)

# first image
sample_data[0][0].resize([300, 300])
# first image class
sample_data.classes[sample_data[0][1]]

# first transformed image
to_img(val_data[0][0]).resize([300, 300])

with torch.no_grad():
    logits = model(val_data[0][0].unsqueeze(0))
    predicted_class = logits.argmax(dim=1).item()

print("Predicted Class:", train_data.classes[predicted_class])
```

![Vision Inference](vision-inference.png)

If you want, run a few more tests:

```py
sample_data[1][0].resize([300, 300])
sample_data.classes[sample_data[1][1]]
with torch.no_grad():
    logits = model(val_data[1][0].unsqueeze(0))
    predicted_class = logits.argmax(dim=1).item()
print("Predicted Class:", train_data.classes[predicted_class])

sample_data[2][0].resize([300, 300])
sample_data.classes[sample_data[2][1]]
with torch.no_grad():
    logits = model(val_data[2][0].unsqueeze(0))
    predicted_class = logits.argmax(dim=1).item()
print("Predicted Class:", train_data.classes[predicted_class])
```

You might also want to test the model with your own picture. For that, first upload a 32x32 image to the directory where you run the Docker container with this notebook. Then run the following Python code:

```py
img = Image.open('sample_image.jpg')
input_tensor = transform_train(img).unsqueeze(0)
with torch.no_grad():
    logits = model(input_tensor)
    predicted_class = logits.argmax(dim=1).item()
print("Predicted Class:", train_data.classes[predicted_class])
```

Does the classification work? Let's measure!

```py
trainer.test(model, val_loader)
```

Test loss between 1.2 - 1.4 is expected.

Test loss around 2.3 for the prediction among 10 classes would mean that the model is answering randomly, it is not trained at all.

For improving results, run training in many iterations:

```py
trainer = Trainer(max_epochs=10, devices=1, accelerator="auto")
trainer.fit(model, train_loader, val_loader)
trainer.test(model, val_loader)
```

What does the test loss look now?

Since training can be time consuming, perhaps you would like to save the results, the weights of the model after training so you can continue using them even after shutting down the container. Use the following commands:

```py
# Save the model
trainer.save_checkpoint("cifar10_model00.ckpt")

# Load the model
model = ImageClassifier.load_from_checkpoint("cifar10_model00.ckpt")
```
