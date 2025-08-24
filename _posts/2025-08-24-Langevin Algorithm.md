---
layout: post
title: "Paper Review : Unadjusted Langevin Algorithm"
categories: Deep-Learning
tags:
  - Deep-Learning
  - paper-review
math: true
---
### Background
#### De-noisingAutoencoder
Input Data $x$에 대해 Gaussian Noise $y=x+\epsilon$ 을 입력으로 받아, clean data $x$로 복원하는 방식의 Autoencoder

###### Mainfold Hypothesis
0~9 등의 숫자 손글씨 데이터를 생각해보자. 모든 시각 이미지 데이터를 2차원 벡터로 나타낼 수 있다면, 해당 2차원 벡터 공간에서 손글씨 데이터에 대한 벡터들이 위치하는 곳은 일부에 불과할  것이다. **(부분공간 manifold)**

##### Tweedie's Formula
> $s_\theta(x)=x+\sigma^2\nabla\log{p_\sigma}(x)$

위의 공식은, 어떤 주어진 $x$가 정규 분포를 따를 때 해당 분포의 평균을 구하는 공식이다. 

---

주어진 Denoising Autoencoder는 $\hat{x}=f_\theta (y)$으로 표현할 수 있는데, 이때 $\hat{x}$은 여전히 원본 $x$와 비교하면 noise를 가지고 있다고 볼 수 있다.

Loss function은 간단하게 $\mathcal{L}={\left\lVert \hat{x} - x \right\rVert}^2$와 같은 L2 Norm으로 나타낼 수 있으며 이를 통해 Denoiser를 학습시킬 수 있다.

이때 sample $N$개에 대해선 다음과 같이 기술할 수 있다.
$$\mathcal{L}_{total}={1\over N}\sum_{i=1}^N{\left\lVert \hat{x}_i - x_i \right\rVert}^2$$
이를 최소화 시키는 것은 곧 $\arg\min_{\hat{x}}\mathbb{E}[{\left\lVert \hat{x} - x \right\rVert}^2 | y]$을 푸는 것과 동일하다. ($\hat{x}=x$일 때 최소)
이때 $\mathbb{E}[{\left\lVert \hat{x} - x \right\rVert}^2|y]=\mathbb{E}[(\hat{x}-x)^\intercal (\hat{x}-x)|y]=\left\lVert \hat{x}\right\rVert^2-2\hat{x}^\intercal \mathbb{E}[x|y]+\mathbb{E}[\left\lVert x\right\rVert^2|y]$이 성립한다.

우리는 주어진 값을 최소화 시키고 싶고, 이때 주어진 함수가 $\hat{x}$에 대해 미분 가능하다는 가정 하에, 위를 미분하여 정리하면 
$$2\hat{x}-2\mathbb{E}[x|y]=0\iff\hat{x}=\mathbb{E}[x|y]$$


![MAP](/assets/images/map.png)

주어진 x와 y 모두 정규 분포를 따른다고 가정 시, 주어진 y에 대해 가장 높은 확률의 x에 대한 추정치인 **Maximum A Posteriori Esimate (MAP)** 이 위의 최적화 문제의 해이다.

하지만 만약에 Prior distribution이 저런 Gaussian Distribution이 아닌 Mixture of Gaussian 형태라고 한다면 오히려 Denoiser는 평균을 학습하며 blurry 한 image인 $\hat{x}$을 출력하게 된다.

![MAP](/assets/images/mog.png)

Posterior Mean이 비록 평균적으로는 좋은 추정치이기는 하나, 개별 output 관점에서 보았을 때는 항상 최적의 해는 아닐 수 있다. 그렇다면 어떻게 이 문제를 해결해야 할까?

앞서 살펴본 Tweedie's Formula를 보자.
$\mathbb{E}[x|y]=y+\sigma^2\nabla\log p(y)$

이에 대한 증명은 다음과 같다.

##### Proof : Tweedie's Formula
---
###### 1. Marginalize the density
- $\nabla\log p(y)={\nabla p(y)\over p(y)}$
- $\nabla p(y)=\nabla\int p(y|x)p(x)dx=\int\nabla p(y|x)p(x)dx with some assumption$

###### 2. Gaussian Likelihood
We know that $p(y|x)\propto \exp(-{1\over 2\sigma^2}\left\lVert y - x \right\rVert)$ so that it would not be difficult to differentiate.
The result is that $\nabla p(y|x)=-{1\over\sigma^2}(y-x)p(y|x)$

Thus
$\nabla p(y)=\int -{1\over\sigma^2}(y-x)p(y|x)p(x)dx=-{1\over\sigma^2}y\int p(y|x)p(x)dx+{1\over\sigma^2}\int p(y|x)xp(x)dx$
which can be shortened as $\nabla\log p(y)=-{1\over\sigma^2}(y-\mathbb{E}[x|y])$

---
![MAP](/assets/images/compass.png)
이때 Tweedie's formula는 noisy image를 가장 높은 확률 분포로 이끄는 de-noising step으로 이해할 수 있고, 이를 반복하여 적용시켜, 확률 분포 상의 local minima로 수렴(gradient ascent의 일종) 시킬 수 있다.

하지만 unknown distribution으로부터 direct 하게 score를 계산하는 것은 어려우므로 denoising autoencoder를 일종의 score estimate로 활용

$$x_{t+1} = x_t +{\eta\over\sigma^2}(s_\theta(x_t)-x_t)$$

이와 같은 과정을 반복하여 가장 높은 확률을 가지는 극대점으로 x를 옮겨 데이터를 생성할 수 있다.
하지만 이 경우 극대점 한 곳에만 수렴하므로, 생성된 데이터의 다양성을 위해 random term을 두게 된다.

$$x_{t+1}=x_t+{\eta\over\sigma^2}(s_theta(x_t)-x_t)+\sqrt{2\eta}z\,\,,\,\,z\sim N(0,I)$$

이러한 랜덤 변수 z를 통해 한 곳에 수렴하지 않고, local minima 주변부로 모이게 되는 효과를 볼 수 있다.

![MAP](/assets/images/comparison_langevin.png)

이를 통해 원본 이미지에 가까운 샘플들을 뽑을 수 있게 된다.
(파란색 : 생성된 이미지, 주황색 : 원본 이미지)

---
#### Limitations
Random image로부터 denoising을 통해 이미지를 생성하는 거지만, 만약 데이터셋에서 low density region, 즉 데이터 샘플이 거의 없는 구역에서 랜덤 이미지가 생성 되고 출발한다면, 유의미한 score estimate를 얻을 수 없으므로 계속해서 노이즈만 쌓여 무의미한 이미지가 생성된다.

이를 해결한 것이 Diffusion Model에 해당한다.