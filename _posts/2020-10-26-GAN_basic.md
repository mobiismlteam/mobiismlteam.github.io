---
layout: article
title: GAN Basic
mathjax: true
toc : true
tags : GAN
---
# GAN
## Definition
- GANs : Generative Adversarial Networks
  - Generative : 생성의
  - Adversarial : 대립 관계의, 적대적인

- Generator(생성자) 와 Discriminator(감별자) 두 개의 네트워크로 구성되어 있음
  - ![](https://3.bp.blogspot.com/-BgYz6OQc4WU/WchaisOCgOI/AAAAAAAACI0/ONloRtdmVisug_HbkotMbP9tr2hkyfg-ACK4BGAYYCw/s1600/kakao_report2.png)
  - 일반적인 비유 : 지폐 위조범과 경찰
    - 위조범은 더욱 진짜같은 가짜 위조지폐를 만드려고 한다 : Generator
    - 경찰은 지폐 감별능력을 높여 위조지폐를 잡아야 한다 : Discriminator  

- 최종 : 위조범의 능력이 정점에 달하면, 경찰은 진짜와 위조 지폐를 찍어서 맞추는 수밖에 없다.($D(x)=\frac{1}{2},p=0.5$)
  - 경찰의 능력이 좋아지면, 위조범은 더 정밀한 위폐를 만들어야한다. -> 위조범의 능력이 좋아지면 경찰도 감별능력을 높여야한다. -> 피드백
  - 실제로는 이렇게 이상적으로 돌아가지 않는다..
    - mode-collapse : 위조범이 1000원만 기가막히게 만들었고 경찰이 구분을 못한다. -> 계속 1000원만 만들것이다.
      
## Equations
- $x$ : sample from real data
- $z \sim p_z$ : latent variable ($p_z$ : generally gaussian)
- $G(z) \sim p_g$
- objective
  - Generator : $p_g=p_{data}$
  - Discriminator : $D(x)=1, D(G(z))=0$
  $$
  V(D,G) = \mathbb{E}_{x\sim p_{data}(x)}[\log{D(x;\theta_D)}]+
           \mathbb{E}_{z\sim p_{z}(z)}[\log{(1-D(G(z;\theta_G);\theta_D)}]
  $$
  - D가 완벽할 때
    - $x \sim p_{data}(x), D(x)=0 \implies 1st~term=0$
    - $z \sim p_{z}(z), D(G(z))=0 \implies 2nd~term=0$
    - $\therefore V(D,G)=0 \implies \max_{D} V(D,G)$
    - D의 목적은 V 값의 최대화
  
  - G가 완벽할 때
    - 1st term 은 z와 관계없으므로 constant
    - $z \sim p_{z}(z), D(G(z))=1 \implies 2nd~term=-\infty$
    - $\therefore V(D,G)=-\infty \implies \min_{G} V(D,G)$
    - G의 목적은 V값의 최소화    
  - re-cap objective : find $G,D$ which satisfy $\min_{G} \max_{D} V(D,G)$

- 임의의 G에 대한 최적의 D : $D^{*}_G(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$
- $\max_D V(D,G) = C(G) = -\log(4) + 2 \cdot JSD(p_{data}||p_g)$
- re-cap objective : $\min_G C(G) : \min_G JSD(p_{data}||p_g)$
  - JSD(Jensen Shannon Divergence) : sum of commuted KL-divergence
  - JSD 측도를 최소화하는 문제와 동일
  - G는 임의 형태의 pdf -> non-parametric 한 G를 만들기 위해 NN을 사용 -> GANs


## Pros/Cons
- GAN : sampler
  - 최적화시에 $p_g = p_{data}$ 가 되도록하는 $G(z) \sim p_g$을 만든다.
    - $z$의 mapping function 을 구하는 것이 목적이므로, $x \sim p_{data}$ 를 직접 구하는 것과는 다르다. : 데이터 분포를 직접 구하는 것이 아니다.
    - 따라서, 데이터 분포 자체를 구하기 위해 tracktable likelihood 를 가정하는 다른 모델과 다르게, likelihood-free 하다.

- GAN 학습의 어려움
  - Convergence
    - D,G 를 동시에 구한다 -> saddle point 를 고려하지 않으면 학습이 영원이 이루어지지 않을 수 있다.
  - mode collapse
    - NN으로 푸는 현실적인 GAN의 경우, 매 단계마다 최적의 $D^{*}$ 를 구할 수 없다. 따라서, value function 을 G와 D에 대해 번갈아가면서 풀어야한다.
      - $G^{*} = \min_{G} \max_{D} V(G,D)$
      - G,D 에 대해 번갈아가며 풀경우, 위 식은 $G^{*} = \max_{D} \min_{G} V(G,D)$ 와 다르지 않다.
        - $\min_{G}$ 를 먼저 푼다. : D가 가장 헷갈려 할만한 샘플 하나만 만들면 땡 -> 쉬운것만 만드는 G가 된다. -> latent variable $z$에 대한 변화가 크지 않은 $G(z)$가 만들어진다.


## Training
- $\max_{D} V(G,D)$ 를 먼저 푼다.
  - 고정된 G를 두고 다음과 같은 데이터를 D에게 제공한다.
    - 생성기 데이터와 라벨 : (G(z), 0.0)
    - 진짜 데이터와 라벨 : (x,1.0)
    - Discriminator 에만 back-prop
    - Loss function : $L_D (\theta_G, \theta_D) = -V(G,D) = - \mathbb{E}_{x\sim p_{data}(x)}[\log{D(x;\theta_D)}] - \mathbb{E}_{z\sim p_{z}(z)}[\log{(1-D(G(z;\theta_G);\theta_D)}]$ : binary crossentropy 와 동일함
    - $\theta_D$ 만 업데이트
- 고정된 D를 두고 G를 업데이트 한다
  - Loss function : $L_G (\theta_G, \theta_D) = - \mathbb{E}_{z\sim p_{z}(z)}[\log{(1-D(G(z;\theta_G);\theta_D)}]$ 
  - 위 식으로 풀면($G=arg\min_{G}L_G$) G가 처음에 만드는 G(z)는 당연히 이상할 확률이 높으므로, 학습 초기에 값이 잘 변하지 않는다.
  - 따라서, 함수를 $L_G (\theta_G, \theta_D) = - \mathbb{E}_{z\sim p_{z}(z)}[\log{D(G(z;\theta_G);\theta_D)}]$ 로 바꾸어, $G=arg\max_{G}L_G$ 찾는 문제로 변경한다.
  - 이후 $\theta_G$ 만 업데이트

## Example
- DCGAN(Deep-convolutional GAN)
  - Maxpooling, Upsampling 대신 strides>1 convolution 을 사용하여 feature map 크기를 조정하는 방법을 학습하게 함
  - Dense 는 z 받을때만
  - Batch normalization 을 하지만, G 출력과 D 입력에는 사용
  - Generator에서는 출력에 tanh(MNIST는 sigmoid), 나머지는 ReLU
  - Discriminator에서는 전무 Leaky ReLU, alpha=0.2