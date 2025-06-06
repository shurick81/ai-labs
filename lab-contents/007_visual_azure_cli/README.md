# Image recognition using a Cloud VM

When a task is too resource consuming for running on your laptop, you can hire a VM in the cloud. There might be a few options.

## Using Azure Cloud Shell

In Azure there are a few VM sizes you can choose from when working with ML:

- Standard_NC6s_v3, Nvidia Tesla V100 GPU (16GB), minumum VRAM is 16, default quota is 0
- Standard_NC4as_T4_v3, Nvidia Tesla T4 GPU (16GB), minumum VRAM is 16, default quota is 0
- Standard_ND6s, Nvidia Tesla P40 GPU (24GB), minimum VRAM is 24, default quota is 0
- Standard_NG8ads_V620_v1, AMD Radeon PRO V620 GPU (32GB), 8-32 GB VRAM, default quota (StandardNGADSV620v1Family) is 0
- Standard_NV12s_v3, Nvidia Tesla M60 GPU (16GB), 8-32 GB VRAM, default quota (standardNVSv3Family) is 0
- Standard_NV4as_v4, AMD Instinct MI25 GPU (16GB), minimum VRAM is 2, by default quota is 8, only supporting Windows
- Standard_NV4ads_V710_v5, AMD Radeon™ Pro V710 (24GB), minimum VRAM is 4, default quota (StandardNVadsV710v5Family) is 0
- Standard_NV6ads_A10_v5, Nvidia A10 GPU (24GB), minimum VRAM is 2, by default quota is 0
- Standard_NC24ads_A100_v4, Nvidia PCIe A100 GPU (80GB), minimum VRAM is 80, 
- Standard_ND96isr_H200_v5, Nvidia H200 GPU (141GB), 1128 GB VRAM, default quota (standardNDISRH200V5Family) is 0
- Standard_ND96isr_MI300X_v5, AMD Instinct MI300X GPU (192GB), 1535 GB VRAM, default quota is 0

https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview?tabs=breakdownseries%2Cgeneralsizelist%2Ccomputesizelist%2Cmemorysizelist%2Cstoragesizelist%2Cgpusizelist%2Cfpgasizelist%2Chpcsizelist#gpu-accelerated

### Nvidia

Before starting, make sure that in your Azure subscription you have at least 64 Cores quota available for `Standard NCASv3_T4 Family vCPUs` (`standardNCASv3_T4Family`). This itself can take awhile.

```bash
# Create ~/.ssh directory if it does not exist
mkdir -p ~/.ssh

# Create an SSH key pair
ssh-keygen -m PEM -t rsa -b 4096 -f ~/.ssh/id_rsa.pem
cat ~/.ssh/id_rsa.pem.pub

az group create --name ai-labs-00 --location westus3
az vm delete \
  --resource-group ai-labs-00 \
  --name small_nvidia \
  -y;
start_time=$(date +%s.%N)  # Capture start time
az vm create \
  --resource-group ai-labs-00 \
  --name small_nvidia \
  --image Microsoft-DSVM:Ubuntu-HPC:2204:22.04.2024102301 \
  --size Standard_NC4as_T4_v3 \
  --security-type Standard \
  --os-disk-size-gb 256 \
  --os-disk-delete-option Delete \
  --admin-username admin98475897 \
  --ssh-key-values ~/.ssh/id_rsa.pem.pub;
end_time=$(date +%s.%N)  # Capture end time
elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
echo "Elapsed time: $elapsed_time seconds"
# 65-85 sec

vm_ip_address=$(az vm list-ip-addresses --resource-group ai-labs-00 --name small_nvidia --query [].virtualMachine.network.publicIpAddresses[].ipAddress -o tsv);
ssh-keygen -R $vm_ip_address;
az ssh vm --resource-group ai-labs-00 --name small_nvidia --local-user admin98475897 --private-key-file ~/.ssh/id_rsa.pem "echo test";
```

confirm `yes`
then run

```bash
start_time=$(date +%s.%N)  # Capture start time
az ssh vm --resource-group ai-labs-00 --name small_nvidia --local-user admin98475897 --private-key-file ~/.ssh/id_rsa.pem "sudo usermod -aG docker admin98475897";
az ssh vm --resource-group ai-labs-00 --name small_nvidia --local-user admin98475897 --private-key-file ~/.ssh/id_rsa.pem docker pull nvcr.io/nvidia/pytorch:25.02-py3;
az ssh vm --resource-group ai-labs-00 --name small_nvidia --local-user admin98475897 --private-key-file ~/.ssh/id_rsa.pem "docker run --rm --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -benchmark";
end_time=$(date +%s.%N)  # Capture end time
elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
echo "Elapsed time: $elapsed_time seconds"
# 770-775 sec
```

expected output:

```
> Compute 7.5 CUDA device: [Tesla T4]
40960 bodies, total time for 10 iterations: 106.236 ms
= 157.924 billion interactions per second
= 3158.482 single-precision GFLOP/s at 20 flops per interaction
```

taking image and deleting the VM:

```bash
start_time=$(date +%s.%N)  # Capture start time
az ssh vm --resource-group ai-labs-00 --name small_nvidia --local-user admin98475897 --private-key-file ~/.ssh/id_rsa.pem "sudo waagent -deprovision+user -force";
az vm deallocate --resource-group ai-labs-00 --name small_nvidia;
az vm generalize --resource-group ai-labs-00 --name small_nvidia;
az image create \
  --resource-group ai-labs-00 \
  --name small_nvidia_00 \
  --source small_nvidia \
  --os-type Linux \
  --hyper-v-generation V2;
az vm delete \
  --resource-group ai-labs-00 \
  --name small_nvidia \
  -y;
end_time=$(date +%s.%N)  # Capture end time
elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
echo "Elapsed time: $elapsed_time seconds"
# 100-105 sec
```

```bash
start_time=$(date +%s.%N)  # Capture start time
image_id=$(az resource show --resource-group ai-labs-00 --name small_nvidia_00 --resource-type Microsoft.Compute/images --query id -o tsv);
az vm create \
  --resource-group ai-labs-00 \
  --name small_nvidia \
  --image $image_id \
  --size Standard_NC4as_T4_v3 \
  --security-type Standard \
  --os-disk-size-gb 256 \
  --os-disk-delete-option Delete \
  --admin-username admin98475897 \
  --ssh-key-values ~/.ssh/id_rsa.pem.pub;
end_time=$(date +%s.%N)  # Capture end time
elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
echo "Elapsed time: $elapsed_time seconds"
# 65-70 sec
vm_ip_address=$(az vm list-ip-addresses --resource-group ai-labs-00 --name small_nvidia --query [].virtualMachine.network.publicIpAddresses[].ipAddress -o tsv);
ssh-keygen -R $vm_ip_address;
az ssh vm --resource-group ai-labs-00 --name small_nvidia --local-user admin98475897 --private-key-file ~/.ssh/id_rsa.pem "sudo usermod -aG docker admin98475897";
```

confirm `yes`
then run

```bash
start_time=$(date +%s.%N)  # Capture start time
az ssh vm --resource-group ai-labs-00 --name small_nvidia --local-user admin98475897 --private-key-file ~/.ssh/id_rsa.pem "
cat <<EOF | sudo tee Dockerfile
FROM nvcr.io/nvidia/pytorch:25.02-py3
RUN pip install lightning==2.5.0
RUN pip install lightning[extra]
EOF
docker buildx build --platform=linux/amd64 --progress=plain --no-cache . -t nvcr.io-nvidia-pytorch-25.02-py3-lightning-2.5.0"
end_time=$(date +%s.%N)  # Capture end time
elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
echo "Elapsed time: $elapsed_time seconds"
# 140-150 sec

start_time=$(date +%s.%N)  # Capture start time
az ssh vm --resource-group ai-labs-00 --name small_nvidia --local-user admin98475897 --private-key-file ~/.ssh/id_rsa.pem "sudo waagent -deprovision+user -force";
az vm deallocate --resource-group ai-labs-00 --name small_nvidia;
az vm generalize --resource-group ai-labs-00 --name small_nvidia;
az image create \
  --resource-group ai-labs-00 \
  --name small_nvidia_01 \
  --source small_nvidia \
  --os-type Linux \
  --hyper-v-generation V2;
az vm delete \
  --resource-group ai-labs-00 \
  --name small_nvidia \
  -y;
end_time=$(date +%s.%N)  # Capture end time
elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
echo "Elapsed time: $elapsed_time seconds"
# 100-140 sec
```

```bash
start_time=$(date +%s.%N)  # Capture start time
image_id=$(az resource show --resource-group ai-labs-00 --name small_nvidia_01 --resource-type Microsoft.Compute/images --query id -o tsv);
az vm create \
  --resource-group ai-labs-00 \
  --name small_nvidia \
  --image $image_id \
  --size Standard_NC64as_T4_v3 \
  --security-type Standard \
  --os-disk-size-gb 256 \
  --os-disk-delete-option Delete \
  --admin-username admin98475897 \
  --ssh-key-values ~/.ssh/id_rsa.pem.pub;
end_time=$(date +%s.%N)  # Capture end time
elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
echo "Elapsed time: $elapsed_time seconds"
# 65-115 sec
vm_ip_address=$(az vm list-ip-addresses --resource-group ai-labs-00 --name small_nvidia --query [].virtualMachine.network.publicIpAddresses[].ipAddress -o tsv);
ssh-keygen -R $vm_ip_address;
az ssh vm --resource-group ai-labs-00 --name small_nvidia --local-user admin98475897 --private-key-file ~/.ssh/id_rsa.pem "sudo usermod -aG docker admin98475897";
```

confirm `yes`
then run

```bash
az ssh vm --resource-group ai-labs-00 --name small_nvidia --local-user admin98475897 --private-key-file ~/.ssh/id_rsa.pem '
cat <<EOF | sudo tee learner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics.classification import Accuracy

pl.seed_everything(42)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
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

train_loader = DataLoader(
    train_data,
    batch_size=64,         # Increased batch size
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

# Step 3: Define the Image Classification Model
class ImageClassifier(pl.LightningModule):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)
    
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
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        test_loss = F.nll_loss(outputs, labels)
        acc = self.test_accuracy(outputs, labels)  # <- Accuracy here
        self.log("test_loss", test_loss, sync_dist=True)
        self.log("test_accuracy", acc, sync_dist=True)
        return {"loss": test_loss, "accuracy": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.004)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

# Initialize the model
model = ImageClassifier()

# Create a Trainer object
trainer = Trainer(max_epochs=25, devices=4, accelerator="gpu")

transform_test = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)
val_loader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=8)

trainer.fit(model, train_loader)
trainer.test(model, val_loader)

#Save the training results in file:
trainer.save_checkpoint("cifar10_model00.ckpt")
'
az ssh vm --resource-group ai-labs-00 --name small_nvidia --local-user admin98475897 --private-key-file ~/.ssh/id_rsa.pem 'docker run --rm -v $PWD:/usr/src --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io-nvidia-pytorch-25.02-py3-lightning-2.5.0 python /usr/src/learner.py'

# copy the checkpoint file from the VM to Azure Cloud Shell
vm_ip_address=$(az vm list-ip-addresses --resource-group ai-labs-00 --name small_nvidia --query [].virtualMachine.network.publicIpAddresses[].ipAddress -o tsv);
az ssh vm --resource-group ai-labs-00 --name small_nvidia --local-user admin98475897 --private-key-file ~/.ssh/id_rsa.pem "sudo chown admin98475897:admin98475897 /home/admin98475897/cifar10_model00.ckpt"
scp -i ~/.ssh/id_rsa.pem admin98475897@$vm_ip_address:/home/admin98475897/cifar10_model00.ckpt .
```

```bash
# Back from the Azure Cloud Shell to the VM:
vm_ip_address=$(az vm list-ip-addresses --resource-group ai-labs-00 --name small_nvidia --query [].virtualMachine.network.publicIpAddresses[].ipAddress -o tsv);
scp -i ~/.ssh/id_rsa.pem cifar10_model00.ckpt admin98475897@$vm_ip_address:/home/admin98475897/
```

Treating the VM as a disposable compute resource, delete it as soon as you don't need compute power, so it does not cost you more than necessary:

```bash
az vm delete \
  --resource-group ai-labs-00 \
  --name small_nvidia \
  -y;
```

In some cases, when you have files that only saved on the VM, you might want to stop it without deleting:

```bash
az vm deallocate \
  --resource-group ai-labs-00 \
  --name small_nvidia;
```

Here's some example of the time that it takes to train a model:

- Hardware: Standard_NC64as_T4_v3 Azure VM, 4 x T4 Nvidia Tesla T4 GPU (16GB)
- Epochs: 25
- Time taken: 26 sec
- Prediction Accuracy: 0.65
- Test Loss: 0.95
