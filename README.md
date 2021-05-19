# Visual Transformer-Review

## 0 导读

初级理解：
- [`《3万字长文带你轻松入门视觉Transformer》`](https://zhuanlan.zhihu.com/p/308301901): 推荐仔细阅读，里面有些小笔误，但是问题不大
- [`《Transformer升级之路：Sinusoidal位置编码追根溯源》`](https://zhuanlan.zhihu.com/p/359500899): 关于位置编码的深入解析与推导
- [`《The Annotated Transformer》`](http://nlp.seas.harvard.edu/2018/04/03/attention.html): harvard 复现attention is all you need 代码

高层次理解：
- [`《How Transformers work in deep learning and NLP: an intuitive introduction》`](https://theaisummer.com/transformer/): 高层次理解
- [`《The Illustrated Transformer》`](http://jalammar.github.io/illustrated-transformer/): 图解 transformer
- [`《The Transformer Family》`](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html): transformer 的技术摘要，最新发展的总结

应用：
- [`《Transformers》`](https://github.com/huggingface/transformers)：transformer 在不同领域的应用，GitHub 项目，star 高达 42k
- 《搞懂视觉Transformer原理和代码，看这篇技术综述就够了！》[`part 1`](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA%3D%3D&chksm=ec1c9073db6b1965d69cdd29d40d51b0148121135e0e73030d099f23deb2ff58fa4558507ab8&idx=1&mid=2247531914&scene=21&sn=3b8d0b4d3821c64e9051a4d645467995#wechat_redirect) [`part 2`](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA%3D%3D&chksm=ec1ca0cbdb6b29dd154fc54e28689dd9b074183920e9d7e0b675d9f9eb7946b8e45b94ab28d3&idx=1&mid=2247535922&scene=21&sn=f4ea9fcee78ac604c03924e367844e85#wechat_redirect) [`part 3`](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA%3D%3D&chksm=ec1ccf34db6b4622badf7244b6ef6c8809b20a1c60faefb3937822f087f93d27ed019b45907c&idx=1&mid=2247541453&scene=21&sn=f9dfe3bcf5e85b413ce1543178681e1e#wechat_redirect) [`part 4`](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA%3D%3D&chksm=ec1cdb48db6b525efa3b087a3798934b6c9082741d41c3d5efe4c74b4e2b470ccaefec3eb787&idx=1&mid=2247546545&scene=21&sn=d9cc512b88c89b8a00482d13250cfd49#wechat_redirect) ：transformer 在不同领域的几个具体应用


## 1 论文和代码

#### 源头论文
- [2017-NIPS-Google] Attention is all you need [`论文`](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

#### 综述型论文
- [2021-Arxiv] A survey on Visual Transformer [`论文`](https://arxiv.org/pdf/2012.12556.pdf)
- [2021-Arxiv] Transformers in Vision: A Survey [`论文`](https://arxiv.org/pdf/2101.01169.pdf)

#### 重要论文
- [2018-NAACL-Google] [自然语言处理] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [`论文`](https://arxiv.org/pdf/1810.04805.pdf) [`代码`](https://github.com/google-research/bert)
- [2020-Arxiv-OpenAI] [自然语言处理] Language Models are Few-Shot Learners (GPT-3) [`论文`](https://arxiv.org/pdf/2005.14165.pdf) [`代码`](https://github.com/openai/gpt-3)
- [2020-ECCV-Facebook] [目标检测] End-to-End Object Detection with Transformers(DERT) [`论文`](https://arxiv.org/pdf/2005.12872.pdf) [`代码`](https://github.com/facebookresearch/detr)
- [2020-ICML-OpenAI] [图像分类] Generative Pretraining from Pixels(IGPT) [`论文`](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf) [`代码`](https://github.com/openai/image-gpt)
- [2021-ICLR-Google] [图像分类] An Image Is Worth 16X16 Words: Transformers for Image Recognition At Scale(ViT) [`论文`](https://arxiv.org/pdf/2010.11929.pdf) [`代码`](https://github.com/google-research/vision_transformer)
- [2020-Arxiv] [数据增强] Pre-Trained Image Processing Transformer(IPT) [`论文`](https://arxiv.org/pdf/2012.00364.pdf)
- [2020-Arxiv-Facebook] [图像分类] Training data-efficient image transformers & distillation through attention(DeiT) [`论文`](https://arxiv.org/pdf/2012.12877.pdf)
- [2021-CVPR-Fudan] [语义分割] Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers(SETR) [`论文`](https://arxiv.org/pdf/2012.15840.pdf) [`代码`](https://github.com/fudan-zvg/SETR)
- [2021-Arxiv-Nvidia] [医疗语义分割] UNETR: Transformers for 3D Medical Image Segmentation [`论文`](https://arxiv.org/pdf/2103.10504.pdf) [`代码`](https://github.com/jeya-maria-jose/Medical-Transformer)
- [2021-Arxiv-Microsoft] [语义分割] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [`论文`](https://arxiv.org/pdf/2103.14030.pdf) [`主代码`](https://github.com/microsoft/Swin-Transformer) [`目标检测代码`](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) [`语义分割代码`](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) 
- [2021-CVPR-TUM] [医疗语义分割] Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation [`论文`](https://arxiv.org/pdf/2105.05537.pdf) [`代码`](https://github.com/HuCaoFighting/Swin-Unet)
- [2021-Arxiv] [医疗语义分割] TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation [`论文`](https://arxiv.org/pdf/2102.04306.pdf) [`代码`](https://github.com/Beckschen/TransUNet)
- [医疗语义分割] Medical Transformer: Gated Axial-Attention for Medical Image Segmentation [`论文`](https://arxiv.org/pdf/2102.10662.pdf) [`代码`](https://github.com/jeya-maria-jose/Medical-Transformer)

#### 其他论文
- [2020-Arxiv] Efficient Attention: Attention with Linear Complexities [`论文`](https://arxiv.org/pdf/1812.01243.pdf) [`代码`](https://github.com/cmsflash/efficient-attention)
- [2017-facebook] Non-local Neural Networks [`论文`](https://arxiv.org/pdf/1711.07971.pdf) [`代码`](https://github.com/facebookresearch/video-nonlocal-net)

#### 数据集
- [目标检测，语义理解] ADE20K [`下载`](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- CIFAR-10 [`下载(python version)`](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
- CIFAR-100 [`下载(python version)`](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- MoNuSeG Dataset(不同器官的肿瘤分割-2018) [`下载`](https://monuseg.grand-challenge.org/Data/)
- GLAS Dataset(腺体肿瘤分割-2015) [`下载`](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/)
- MSD Dataset(Medical Segmentation Decathlon) [`下载`](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)

#### reference
https://github.com/sannykim/transformers

## 其他资源
#### Follow-Up Papers
Since the original paper was published, there has been a massive wave of papers building on the Transformer. Most notably, BERT, GPT-2, XLNet and Reformer. 
- [Linformer](https://arxiv.org/abs/2006.04768)
- [Reformer](https://openreview.net/forum?id=rkgNKkHtvB)
- [TransformerXL](https://arxiv.org/abs/1901.02860)
- [Evolved Transformer](https://arxiv.org/abs/1901.11117)
- [Image Transformer](https://arxiv.org/abs/1802.05751)
- [Music Transformer](https://arxiv.org/abs/1809.04281)
- [TTS Transformer](https://arxiv.org/abs/1809.08895)
- [Set Transformer](https://arxiv.org/abs/1810.00825)
- [Sparse Transformer](https://arxiv.org/abs/1904.10509)
- [Levenshtein Transformer](https://arxiv.org/abs/1905.11006)
- [BERT](https://arxiv.org/abs/1810.04805)
- [GPT-1](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3](https://arxiv.org/abs/2005.14165)
- [UniLM](https://arxiv.org/abs/1905.03197)
- [XLNet](https://arxiv.org/abs/1906.08237)
- [MASS](https://arxiv.org/abs/1905.02450)
- [Adapative Attention Spans](https://arxiv.org/abs/1905.07799)
- [All Attention Layers](https://arxiv.org/abs/1907.01470)
- [Large Memory Layers with Product Keys](https://arxiv.org/abs/1907.05242)

#### BERT
- [Jacob Devlin's ICML Talk](https://videoken.com/embed/uN4PKDp5HOU?tocitem=4)
- [AISC](https://www.youtube.com/watch?v=BhlOGGzC0Q0)
- [Yannic Kilcher](https://www.youtube.com/watch?v=-9evrZnBorM)
- [The Illustrated BERT, ELMo, and co.](http://jalammar.github.io/illustrated-bert/)
- [Yashu Seth's BERT FAQ](https://yashuseth.blog/2019/06/12/bert-explained-faqs-understand-bert-working/)
- [Chris McCormick's BERT Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)
- [Chris McCormick's BERT Fine-Tuning Tutorial](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)

#### GPT-1, GPT-2 and GPT-3
- [OpenAI's GPT-1 Blog Post](https://openai.com/blog/language-unsupervised/)
- [OpenAI's GPT-2 Blog Post](https://openai.com/blog/better-language-models/)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Alec Radford's Guest Lecture on Language Models](https://www.youtube.com/watch?v=GEtbD6pqTTE&t=2057s)
- [Yannic Kilcher on GPT-2](https://www.youtube.com/watch?v=u1_qMdb0kYU)
- [Yannic Kilcher on GPT-3](https://www.youtube.com/watch?v=SY5PvZrJhLE)
- [How GPT-3 Works - Visualizations and Animations](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)
- [OpenAI GPT-3 API](https://openai.com/blog/openai-api/)

#### Transformer XL and XLNet
- [AISC Review of Transformer XL](https://www.youtube.com/watch?v=cXZ9YBqH3m0&t=2226s)
- [Microsoft Reading Group on Transformer XL](https://www.youtube.com/watch?v=cXZ9YBqH3m0&t=2226s)
- [Yannic Kilcher](https://www.youtube.com/watch?v=H5vpBCLo74U)
- [NLP Breakfasts' Overview of XLNet](https://www.youtube.com/watch?v=cXZ9YBqH3m0&t=2226s)

更新时间：2021/05/19
