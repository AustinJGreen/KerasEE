# KerasEE
Keras machine learning library applied to Everybody Edits. Here I'm just experimenting with data from user worlds and
its potential to create interesting things.

## World Auto Encoder
One part of this project is aimed towards training an auto encoder. This model is useful for other models as the auto
encoder learns features of worlds quite well. Figure 1 shows a sample of encoded worlds.

<p align="center">
  <img src="https://github.com/ajosg/KerasEE/blob/master/plots/ae_plot.png?raw=true" alt="Whoops, plot is missing."/>
  Figure 1: 4 worlds encoded using a trained auto encoder at a resolution of 112x112.
</p>


## Classifiers
In everybodyedits there are a variety of different worlds. Using labeled data I've trained a few classifiers for a
variety of genres. 

### Pro Classifier
First classifier I trained was a "Pro" classifer. This was trained based upon worlds built by builders with lots of
experience who balance worlds with art and playability and most importantly originality. Yes this is highly subjective.
Worlds with stairs, basic minigames, and the like are not considered pro. However, the classifier was trained based upon
a variety of well built and liked worlds by the community which include pure art levels, pure frustrations, and the
classics. Figure 2 shows a classification done on a sample of unlabeled worlds.
<p align="center">
  <img src="https://github.com/ajosg/KerasEE/blob/master/plots/pro_plot3x3.png?raw=true" alt="Whoops, plot is missing."/>
  Figure 2: 9 samples with varying confidence levels for a "pro" level.
</p>

## World Generater
One of the more challenging prospects I've had in mind is to be able to generate seemingly good small portions of worlds.
As of the current state of the project, I've been generating samples up to 64x64 in size. In order to generate worlds
even close to a regular 200x200 world like 256x256 or even 128x128 I would need more memory on my machine. 

## World Inpainting
WIP
