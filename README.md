# Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation. 
## Chang Liu, Giulia Rizzoli, Pietro Zanuttigh, Fu Li, Yi Niu -- ECCV 2024

#### Official PyTorch Implementation
![headfig](https://github.com/dota-109/Web-WILSS/blob/main/docs/head-fig.png)

Current weakly-supervised incremental learning for semantic segmentation (WILSS) approaches only consider replacing pixel-level annotations with image-level labels, while the training images are still from well-designed datasets. In this work, we argue that widely available web images can also be considered for the learning of new classes. To achieve this, firstly we introduce a strategy to select web images which are similar to previously seen examples in the latent space using a Fourier-based domain discriminator.  Then, an effective caption-driven reharsal strategy is proposed to preserve previously learnt classes. To our knowledge, this is the first work to rely solely on web images for both the learning of new concepts and the preservation of the already learned ones in WILSS. Experimental results show that the proposed approach can reach state-of-the-art performances without using manually selected and annotated data in the incremental steps. 

![method](https://github.com/dota-109/Web-WILSS/blob/main/docs/main-framework.png)
