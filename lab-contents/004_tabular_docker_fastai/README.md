# fastai Tabular Training Using CLI

Imagine that you want to teach a device to turn on a warning signal of a car seatbelt.

- First input to consider could be whether the seatbelt is fastened or not.
- Second input could be whether the car is in motion or not.
- Third input could be whether the destination point is set in the navigation system.

The latter input should not play any role and we want to see that the after we train the model it will not affect the prediction.

The fourth column in the training data is the decision whether the warning signal should be on or off.

1. Using Docker on Linux, WSL or Mac, run:

```bash
docker run --rm -it -v $PWD:/usr/src fastai/fastai:2021-02-11 /bin/bash -c python
```

2. In the container, execute the following Pytnon commands:

```py
# Training
from fastai.tabular.all import *
train_data = [[0,0,0,0],
              [0,1,0,1],
              [1,0,0,1],
              [1,1,0,0],
              [0,0,1,0],
              [0,1,1,1],
              [1,0,1,1],
              [1,1,1,0]]
df = pd.DataFrame(train_data, columns=['input0', 'input1', 'input2', 'output0'])
df.head()
splits = RandomSplitter()(range_of(df))
y_block = CategoryBlock()
to = TabularPandas(df, procs=[FillMissing, Categorify, Normalize], cat_names = ['input0', 'input1', 'input2'], y_names='output0', splits=splits)
dls = to.dataloaders(bs=2)
dls.show_batch()
learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(200)

# Trying the trained model
test_df = pd.DataFrame([[0,1,0]], columns=['input0', 'input1', 'input2'])
dl = learn.dls.test_dl(test_df)
learn.get_preds(dl=dl)
# Expected output/prediction: closer to 1

test_df = pd.DataFrame([[1,0,0]], columns=['input0', 'input1', 'input2'])
dl = learn.dls.test_dl(test_df)
learn.get_preds(dl=dl)
# Expected output/prediction: closer to 1

test_df = pd.DataFrame([[1,1,1]], columns=['input0', 'input1', 'input2'])
dl = learn.dls.test_dl(test_df)
learn.get_preds(dl=dl)
# Expected output/prediction: closer to 0

# Exporting the model/parameters to a file
learn.export('/usr/src/zeros-ones-00.pkl')
```

3. Ctrl + D to exit the container.

4. Verify that the new file, `zeros-ones-00.pkl`, is created:

```bash
ls -lh
```

Check the size of the `zeros-ones-00.pkl` file.

5. Create a new container again:

```bash
docker run --rm -it -v $PWD:/usr/src fastai/fastai:2021-02-11 /bin/bash -c python
```

6. Run the following Python code:

```py
# Load the model that was saved on previous steps
from fastai.tabular.all import *
learn = load_learner('/usr/src/zeros-ones-00.pkl')

# Trying the trained model
test_df = pd.DataFrame([[0,1,0]], columns=['input0', 'input1', 'input2'])
dl = learn.dls.test_dl(test_df)
learn.get_preds(dl=dl)
# Expected output/prediction: closer to 1

test_df = pd.DataFrame([[1,0,0]], columns=['input0', 'input1', 'input2'])
dl = learn.dls.test_dl(test_df)
learn.get_preds(dl=dl)
# Expected output/prediction: closer to 1

test_df = pd.DataFrame([[1,1,1]], columns=['input0', 'input1', 'input2'])
dl = learn.dls.test_dl(test_df)
learn.get_preds(dl=dl)
# Expected output/prediction: closer to 0
```

7. Follow more tutorials from https://docs.fast.ai/tutorial.tabular.html

# fastai Tabular Training Using Jupiter

1. Using Docker on Linux, WSL or Mac, execute fastai container with Jupiter server:

```bash
docker run --rm -p 8888:8888 fastai/fastai:2021-02-11 /bin/bash -c "\
  mkdir -p /opt/notebooks && \
  jupyter notebook \
    --notebook-dir=/opt/notebooks --ip='*' --port=8888 \
    --no-browser --allow-root"
```

2. Follow one of the tutorials. For example, https://docs.fast.ai/tutorial.tabular.html
