# transcribe

A simple distributed speech classification system with convolutional neural networks built on Apache Spark and Tensorflow. This project was part of a university course on big data applications. While we did not cover Tensorflow in the lectures, I wanted to challenge myself and see if I'd be able to train neural networks on a cluster. Tensorflow does not natively support running on a Spark cluster, but there are ways to make it work. The great people over at Yahoo are building and maintaining [TensorflowOnSpark](https://github.com/yahoo/TensorFlowOnSpark), a Tensorflow distribution that brings neural networks to Spark.

As recommended by the TensorflowOnSpark team, I first prototyped the neural network locally with regular Tensorflow before porting it to run on Spark. While processing and training locally on an emulated Spark cluster has worked, I have yet to deploy the system on a proper cluster to truly train it and see how it performs. I hope to get around to deploying the whole thing on AWS in the near future, leveraging my newly acquired [cloud computing knowledge](https://github.com/DanThePutzer/kumo).

![Spark](https://user-images.githubusercontent.com/25454503/87907051-2258b280-ca64-11ea-87ab-432fc59c400c.png)

### Installation & Usage

Before getting started a few things need be set up on your machine. Depending on your environment you might want to use **pip** or **conda** to install the necessary dependencies (some might not be available through conda, use pip to install those).

```python
# Install dependencies with pip
pip install tensorflow tensorflowonspark pyspark findspark librosa soundfile numpy matplotlib tqdm

# Install dependencies with conda
conda install -c conda-forge tensorflow tensorflowonspark pyspark findspark librosa soundfile numpy matplotlib tqdm
```

You also need Jupyter to be able to open the notebook. Again depending on your environment, pick the proper command to install.

```python
pip install juypter
# or
conda install jupyter
```

Additionally, Apache Spark needs to be installed and properly configured on your machine. The setup depends on your operating system. I linked some guides on how to set up Spark on your machine and configure it to run with Python below.
- [Mac/Linux](https://www.sicara.ai/blog/2017-05-02-get-started-pyspark-jupyter-notebook-3-minutes)
- [Windows](https://medium.com/big-data-engineering/how-to-install-apache-spark-2-x-in-your-pc-e2047246ffc3)

Now simply navigate to the root directory of the repo and run ```jupyter notebook``` in the command line (don't forget to activate your environment first if you are using conda).

### Data & Processing

The project is based on the Speech Commands Dataset v0.01 available on [Kaggle](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data). The dataset contains approximately 64,000 samples of 20 different words in mp3 format. All samples are around 1 second long, making the data somewhat easier to work with. My current approach is to generate spectrograms for each audio sample and train a convolutional neural network on said spectrograms as if it was a regular image.

### Planned Progress

The project is currently functional, meaning Spark and TensorflowOnSpark are running and doing their job when tested locally. The performance of the neural network, however, needs some serious improvement and deployment on a proper cluster is something I definitely want to attempt.

#### Upcoming ToDos:

- Reformat code from jupyter prototype to a true Spark job
- Deploy on a real cluster
- Change/tweak CNN structure
- Test on real audio recordings

&nbsp;

![Daniel Putzer, 2020](https://i.ibb.co/LSxTsY3/dan.png "Daniel Putzer, 2020")  
<https://danielputzer.com>
