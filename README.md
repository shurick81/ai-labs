# ai-labs

If you want to run ML tools and their prerequisites/dependencies in your experimentation environment, what would be the shortest path? Here we collected examples of experiments that you can relatively simply execute even without experience in even installing Python.

## Inside an Artificial Neural Network

[lab-contents/001_inside_an_artificial_neural_network](lab-contents/001_inside_an_artificial_neural_network).

## First Machine Learning Experiments

### Ways to Execute ML Training and Inference

First, let's go through some methods of executing the ML processes without preliminary installing prerequisites in your physical environment, like Laptop.

#### Using ML Cloud Providers

Machine learning cloud providers allow you using the most powerful models that might be quite impossible for you to run otherwise.

```mermaid
flowchart TB
  subgraph provider[ML Cloud Provider]
    service[ML Service]
  end
  laptop-->|Calling API|service
```

#### Using Docker

Use container images that already have such preinstalled software as Python, PyTorch, fastai, Pandas, Jupiter, etc.

```mermaid
flowchart TB
    container_image-->|saved to|docker_hub
    subgraph laptop[Your Laptop]
      persistend_files[Persistent Files]
      container[Disposable Container]-->|mounts|persistend_files
    end
    docker_hub[Docker Hub]
    container-->|pulled from|docker_hub
    subgraph container_image[Container Image]
      Python
      PyTorch
      fastai
      Pandas
      Jupiter
    end
```

Prerequisite for using this approach is Docker installed in Mac, Linux or WSL (Windows) environment.

#### Using Lab VMs

When running on a local docker takes too much resources or too much time, an option might be running the load in the cloud.

```mermaid
flowchart TB
  subgraph provider[Cloud Provider]
    VM[GPU-Accelerated VM]
  end
  laptop-->|remotely control|VM
```

Prerequisite for using this approach is having installed tools for remote control of Cloud provider such as Azure.

| Problem Class | Training/Inference     | Environement | ML Toolset                 | Experiment                                         |
| -             | -                      | -            | -                          | -                                                  |
| LLM           | inference              | cloud        | Gemini 1.5                 | [Section](lab-contents/003_llm_cloud_gemini/README.md#trying-llm-google-gemini-15)           |
| LLM           | inference              | cloud        | Gemini 2.0                 | [Section](lab-contents/003_llm_cloud_gemini/README.md#trying-llm-google-gemini-20)           |
| LLM           | prompt with image      | cloud        | Gemini 2.0                 | [Section](lab-contents/003_llm_cloud_gemini/README.md#adding-an-image-to-the-request)        |
| Tabular       | training and inference | docker       | PyTorch, fastai            | [Section](lab-contents/004_tabular_docker_fastai/README.md#fastai-tabular-training-using-cli)     |
| Tabular       | training and inference | docker       | PyTorch, fastai, Jupiter   | [Section](lab-contents/004_tabular_docker_fastai/README.md#fastai-tabular-training-using-jupiter) |
| visual        | training and inference | docker       | PyTorch Lightning, Jupiter | [Page](lab-contents/005_visual_docker_jupyter)     |
| visual        | training and inference | cloud VM     | PyTorch Lightning, Jupiter | [Page](lab-contents/006_visual_azure_jupyter)      |
| visual        | training and inference | cloud VM     | PyTorch Lightning, CLI | [Page](lab-contents/007_visual_azure_cli)      |
