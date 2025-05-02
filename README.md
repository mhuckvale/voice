# Voice
Experiments in voice conversion and speaker dimensions.

# Demonstrate FreeVC with Deep-Speaker embedding trained on Globe sample dataset

This demnstration shows how a voice conversion system can be trained to be drived by acoustic parameters or by principal components of those acoustic parameters.

In this demonstration, the [FreeVC system](https://github.com/OlaWod/FreeVC) is trained to performa voice conversion of speech audio using speaker embeddings computed by the [Deep-Speaker system](https://github.com/philipperemy/deep-speaker).

Deep Speaker was trained using a balanced set of 1000 speakers from the Globe corpus. FreeVC was trained using 5000 male and 5000 female speakers from the Globe corpus and Deep Speaker embeddings for those speakers.

Acoustic parameters were extracted for each of the 10,000 speakers and an MLP Regression model was used to predict the Deep Speaker embedding from the acoustic parameters.

Finally principal components analysis of the acoustic parameters were extracted to be used in the demonstration interface.

The diagram shows how the PCA components, acoustic parameters and speaker embeddings are used with FreeVC:

![Schematic diagram of FreeVC system](/images/freevc-pca.png)

The user interface controls allow you to set the required acoustic parameters; either directly using sliders or indirectly using principal components:

![User interface for Voice Conversion](/images/pca-controls.png)

Click on Go PCA or Go VQ to synthesize utterances using the PCA components or the raw parameters respectively.

Run [Globe PCA Demonstration](https://colab.research.google.com/github/mhuckvale/voice/blob/main/Globe_PCA_Demonstration.ipynb) in COLAB using a GPU runtime.

