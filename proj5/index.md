---
author: "Arnold Cai"
date: "2024.11.18
---
{% include mathjax_support.html %}

# Proj5a: Power of Diffusion Models

Diffusion model shenanigans. Using [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if) diffusion model trained by Stablility AI. This model takes in `64x64` images and produces `64x64` images from its first stage. I did not upsample the images into `256x256` images using the second stage of the model due to lack of google colab credits :(. The first part will be going through the implementation of a diffusion model and its steps while the second part will be implementing some cool results from some fairly recent papers.

## 0.1 Seed + Setup
I used `SEED=501`.

Here some of the ouput images after passing it through stage 1 and 2 UNets of the model. It appears that the number of iterative steps the model takes affects the output image pretty drastically even though the same word embedding was used.

<center>

<img src="../proj5/out/model-10-steps.png">
<p>10 steps</p>

<img src="../proj5/out/model-20-steps.png">
<p>20 steps</p>

<img src="../proj5/out/model-40-steps.png">
<p>40 steps</p>

</center>


## 1.1 Forward Function
We need to implement a function that adds noise to an image. This is achieved with this formula:

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar\alpha_t}\epsilon, \space where \space \epsilon \sim N(0, 1)
$$

We are using a noise generator (or estimation) using a standard normal distirbution $\epsilon$, which can be calculated via `torch.randn_like`, and an `alpha_cumprod` $\bar\alpha_t$ of $t$ step. As $t$ increases, so does the amount of noise added to the image increase.

<center>
    <table>
        <tbody align=center>
            <tr>
                <td>
                    <img src="../proj5/out/campanile.jpg" width=128 height=128>
                    <p align=center>campanile.jpg</p>
                </td>
                <td>
                    <img src="../proj5/out/campanile-noisy-250.jpg" width=128 height=128>
                    <p align=center>t=250</p>
                </td>
                <td>
                    <img src="../proj5/out/campanile-noisy-500.jpg" width=128 height=128>
                    <p align=center>t=500</p>
                </td>
                <td>
                    <img src="../proj5/out/campanile-noisy-750.jpg" width=128 height=128>
                    <p align=center>t=750</p>
                </td>
            </tr>
        </tbody>
    </table>
</center>

## 1.2 Classical Denoising
After noise-ifying images, we can train a diffusion model to estimate denosising processes of these noisified image. That way the diffusion model can later "generate" images by converting random noisy images not on the real image manifold into images related to the input fed into the model.

We will first try to implement a classical denoising method via Gaussian Blur Filtering. In particular, I used `torchvision.trnasforms.functional.gaussian_blur` with a kernel size of `(7, 7)` to implement the blurs on the noisy images. This will pass the noisy image through a low pass filter and therefore get rid of some of the low frequency noise. However, as seen by the results, the Gaussian Blur Filter doesn't get rid of all the noise and it also blurs the original image.

<center>
    <table>
        <tr>
            <td>
                <img src="../proj5/out/campanile-noisy-250.jpg" width=128 height=128>
                <p align=center>t=250</p>
            </td>
            <td>
                <img src="../proj5/out/campanile-noisy-500.jpg" width=128 height=128>
                <p align=center>t=500</p>
            </td>
            <td>
                <img src="../proj5/out/campanile-noisy-750.jpg" width=128 height=128>
                <p align=center>t=750</p>
            </td>
        </tr>
    </table>
</center>

<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/campanile-gaussian-denoised-250.jpg" width=128 height=128>
            <p align=center>blur t=250</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-gaussian-denoised-500.jpg" width=128 height=128>
            <p align=center>blur t=500</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-gaussian-denoised-750.jpg" width=128 height=128>
            <p align=center>blur t=750</p>
        </td>
    </tr>
</table>
</center>

## 1.3 One Step Denoising
We can further improve the denoising by using a pretrained diffusion model to estimate the noise in the new noisy image and then remove that estiamted noise from that same noisy image to get closer towards the original image. Since DeepFloyd was trained on text conditioning, we use the first stage UNet on the condition of `"a high quality photo"`.

In comparison to the Gaussian Blur Filter, this method of denoising gets rid of all the noise. However, the predicted image still tends to be blurred and loeses some of the structure and detailes that were in the original image.

<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/campanile-noisy-250.jpg" width=128 height=128>
            <p align=center>t=250</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-noisy-500.jpg" width=128 height=128>
            <p align=center>t=500</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-noisy-750.jpg" width=128 height=128>
            <p align=center>t=750</p>
        </td>
    </tr>
</table>
</center>

<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/campanile-unet-denoised-250.jpg" width=128 height=128>
            <p align=center>one-step t=250</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-unet-denoised-500.jpg" width=128 height=128>
            <p align=center>one-step t=500</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-unet-denoised-750.jpg" width=128 height=128>
            <p align=center>one-step t=750</p>
        </td>
    </tr>
</table>
</center>

## 1.4 Iterative Denoising

Another method of denoising we can use is iterative denoising, the default denoising method used by diffusion models. It would be tedious and expensive to go through each step, espcially if $T$ is very large. Therefore, we iterate through some `strided_timesteps` with `strides=30`. The formula is given below with $t$ being the current timestep and $t'$ being an earlier timestep such that $t' < t$.

$$
x_{t'} = \frac{\sqrt{\bar\alpha_{t'}}\beta_t}{1 - \bar\alpha_t} x_0 +
        \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t'})}{1 - \bar\alpha_t} x_t +
        v_\sigma
$$

$$
\alpha_t = \frac{\bar\alpha_t}{\bar\alpha_{t'}}
$$

$$
\beta_t = 1 - \alpha_t
$$

$x_0$ is the estimated clean image at each iterative step using the formula used in the forward process with noise $\epsilon$ being the estimated noise from UNet output.

<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/campanile-iterative-t-90.jpg" width=128 height=128>
            <p align=center>t=90</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-iterative-t-240.jpg" width=128 height=128>
            <p align=center>t=240</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-iterative-t-390.jpg" width=128 height=128>
            <p align=center>t=390</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-iterative-t-540.jpg" width=128 height=128>
            <p align=center>t=540</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-iterative-t-690.jpg" width=128 height=128>
            <p align=center>t=690</p>
        </td>
    </tr>
</table>
</center>

<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/campanile.jpg" width=128 height=128>
            <p align=center>campanile.jpg</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-iterative-denoised.jpg" width=128 height=128>
            <p align=center>iterative</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-clean_one_step.jpg" width=128 height=128>
            <p align=center>one-step</p>
        </td>
        <td>
            <img src="../proj5/out/campanile-blur-filtered.jpg" width=128 height=128>
            <p align=center>gaussian blur</p>
        </td>
    </tr>
</table>
</center>

## 1.5 Diffusion Model Sampling
We are going to generate images from scratch by starting the iterative denoising at $T$ timestep (the max timestep) and feeding the model a random noisy image generated via `torch.rand_like` and with the word embedding `"a high quailty photo"`. Here are some samples I genereated using iterative denoising.

<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/random-sample-0.jpg" width=128 height=128>
        </td>
        <td>
            <img src="../proj5/out/random-sample-1.jpg" width=128 height=128>
        </td>
        <td>
            <img src="../proj5/out/random-sample-2.jpg" width=128 height=128>
        </td>
        <td>
            <img src="../proj5/out/random-sample-3.jpg" width=128 height=128>
        </td>
        <td>
            <img src="../proj5/out/random-sample-4.jpg" width=128 height=128>
        </td>
    </tr>
</table>
</center>

## 1.6 Classifier Free Guidance (CFG)
Some of the images generated by iterative denoising seem really random or confusing. To fix this, we will use [Classifier Free Guidance](https://arxiv.org/abs/2207.12598), which uses an conditional and unconditional noise estimate the new noise.

$$\epsilon = \epsilon_u + \gamma(\epsilon_c - \epsilon_u)$$

For these images, I used `"a high quality photo"` for the UNet embedding that would estimate conditional noise and a null prompt of `""` as the unconditional noise. Furthermore, I used $\gamma=7$ when calculating the overall noise estimate.

<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/cfg-random-sample-0.jpg" width=128 height=128>
        </td>
        <td>
            <img src="../proj5/out/cfg-random-sample-1.jpg" width=128 height=128>
        </td>
        <td>
            <img src="../proj5/out/cfg-random-sample-2.jpg" width=128 height=128>
        </td>
        <td>
            <img src="../proj5/out/cfg-random-sample-3.jpg" width=128 height=128>
        </td>
        <td>
            <img src="../proj5/out/cfg-random-sample-4.jpg" width=128 height=128>
        </td>
    </tr>
</table>
</center>

## 1.7 Image to Image Translation
Instead of passing in a randomly generated image, we will pass in a noise-ified image (using `forward(img, t)`) of the original image at different timesteps in order to get the diffusion model to output something similar to the original image we noise-ified.

> **Side Note:** I used a `strided_timesteps` array that ranged from `[990, 0]` with a `stride=30`. When `i_start=0`, `t=990`, which the timestep at which `forward(img, t)` would return the noisiest version of the original image.

### campanile.jpg
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/translate-campanile-1.jpg" width=128 height=128>
            <p align=center>i_start=0</p>
        </td>
        <td>
            <img src="../proj5/out/translate-campanile-3.jpg" width=128 height=128>
            <p align=center>i_start=3</p>
        </td>
        <td>
            <img src="../proj5/out/translate-campanile-5.jpg" width=128 height=128>
            <p align=center>i_start=5</p>
        </td>
        <td>
            <img src="../proj5/out/translate-campanile-7.jpg" width=128 height=128>
            <p align=center>i_start=7</p>
        </td>
        <td>
            <img src="../proj5/out/translate-campanile-10.jpg" width=128 height=128>
            <p align=center>i_start=10</p>
        </td>
        <td>
            <img src="../proj5/out/translate-campanile-20.jpg" width=128 height=128>
            <p align=center>i_start=20</p>
        </td>
    </tr>
</table>
</center>

### nyc.jpg
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/translate-nyc-1.jpg" width=128 height=128>
            <p align=center>i_start=0</p>
        </td>
        <td>
            <img src="../proj5/out/translate-nyc-3.jpg" width=128 height=128>
            <p align=center>i_start=3</p>
        </td>
        <td>
            <img src="../proj5/out/translate-nyc-5.jpg" width=128 height=128>
            <p align=center>i_start=5</p>
        </td>
        <td>
            <img src="../proj5/out/translate-nyc-7.jpg" width=128 height=128>
            <p align=center>i_start=7</p>
        </td>
        <td>
            <img src="../proj5/out/translate-nyc-10.jpg" width=128 height=128>
            <p align=center>i_start=10</p>
        </td>
        <td>
            <img src="../proj5/out/translate-nyc-20.jpg" width=128 height=128>
            <p align=center>i_start=20</p>
        </td>
    </tr>
</table>
</center>

### sf.jpg
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/translate-sf-1.jpg" width=128 height=128>
            <p align=center>i_start=0</p>
        </td>
        <td>
            <img src="../proj5/out/translate-sf-3.jpg" width=128 height=128>
            <p align=center>i_start=3</p>
        </td>
        <td>
            <img src="../proj5/out/translate-sf-5.jpg" width=128 height=128>
            <p align=center>i_start=5</p>
        </td>
        <td>
            <img src="../proj5/out/translate-sf-7.jpg" width=128 height=128>
            <p align=center>i_start=7</p>
        </td>
        <td>
            <img src="../proj5/out/translate-sf-10.jpg" width=128 height=128>
            <p align=center>i_start=10</p>
        </td>
        <td>
            <img src="../proj5/out/translate-sf-20.jpg" width=128 height=128>
            <p align=center>i_start=20</p>
        </td>
    </tr>
</table>
</center>

## 1.7.1 Hand Drawn and Web Images
Let's see if CFG with DeepFloyd runs well on hand drawn images and images taken from the web!

### web: jinx.jpg
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/cfg-web-1.jpg" width=128 height=128>
            <p align=center>i_start=0</p>
        </td>
        <td>
            <img src="../proj5/out/cfg-web-3.jpg" width=128 height=128>
            <p align=center>i_start=3</p>
        </td>
        <td>
            <img src="../proj5/out/cfg-web-5.jpg" width=128 height=128>
            <p align=center>i_start=5</p>
        </td>
        <td>
            <img src="../proj5/out/cfg-web-7.jpg" width=128 height=128>
            <p align=center>i_start=7</p>
        </td>
        <td>
            <img src="../proj5/out/cfg-web-10.jpg" width=128 height=128>
            <p align=center>i_start=10</p>
        </td>
        <td>
            <img src="../proj5/out/cfg-web-20.jpg" width=128 height=128>
            <p align=center>i_start=20</p>
        </td>
        <td>
            <img src="../proj5/out/jinx.jpg" width=128 height=128>
            <p align=center>jinx.jpg</p>
        </td>
    </tr>
</table>
</center>

### hand drawn: pikachu?
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/hand-drawn-pikachu-1.jpg" width=128 height=128>
            <p align=center>i_start=0</p>
        </td>
        <td>
            <img src="../proj5/out/hand-drawn-pikachu-3.jpg" width=128 height=128>
            <p align=center>i_start=3</p>
        </td>
        <td>
            <img src="../proj5/out/hand-drawn-pikachu-5.jpg" width=128 height=128>
            <p align=center>i_start=5</p>
        </td>
        <td>
            <img src="../proj5/out/hand-drawn-pikachu-7.jpg" width=128 height=128>
            <p align=center>i_start=7</p>
        </td>
        <td>
            <img src="../proj5/out/hand-drawn-pikachu-10.jpg" width=128 height=128>
            <p align=center>i_start=10</p>
        </td>
        <td>
            <img src="../proj5/out/hand-drawn-pikachu-20.jpg" width=128 height=128>
            <p align=center>i_start=20</p>
        </td>
                <td>
            <img src="../proj5/out/pikachu.jpg" width=128 height=128>
            <p align=center>pikachu.jpg</p>
        </td>
    </tr>
</table>
</center>

### hand drawn: ditto?
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/hand-drawn-ditto-1.jpg" width=128 height=128>
            <p align=center>i_start=0</p>
        </td>
        <td>
            <img src="../proj5/out/hand-drawn-ditto-3.jpg" width=128 height=128>
            <p align=center>i_start=3</p>
        </td>
        <td>
            <img src="../proj5/out/hand-drawn-ditto-5.jpg" width=128 height=128>
            <p align=center>i_start=5</p>
        </td>
        <td>
            <img src="../proj5/out/hand-drawn-ditto-7.jpg" width=128 height=128>
            <p align=center>i_start=7</p>
        </td>
        <td>
            <img src="../proj5/out/hand-drawn-ditto-10.jpg" width=128 height=128>
            <p align=center>i_start=10</p>
        </td>
        <td>
            <img src="../proj5/out/hand-drawn-ditto-20.jpg" width=128 height=128>
            <p align=center>i_start=20</p>
        </td>
        <td>
            <img src="../proj5/out/ditto.jpg" width=128 height=128>
            <p align=center>ditto.jpg</p>
        </td>
    </tr>
</table>
</center>

## 1.7.2 Inpainting
We can use a mask and only pass in the mask portion through the forwarding process such that the diffusion model will only generate within the masked area.

$$
x_t = \textbf{m} x_t + (1 - \textbf{m})\text{forward}(x_{orig}, t)
$$

### campanile.jpg
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/campanile.jpg" width=128 height=128>
            <p align=center>campanile.jpg</p>
        </td>
        <td>
            <img src="../proj5/out/mask-campanile.jpg" width=128 height=128>
            <p align=center>mask</p>
        </td>
                <td>
            <img src="../proj5/out/replace-campanile.jpg" width=128 height=128>
            <p align=center>to replace</p>
        </td>
        <td>
            <img src="../proj5/out/inpainting-campanile.jpg" width=128 height=128>
            <p align=center>inpainted</p>
        </td>
    </tr>
</table>
</center>

### nyc
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/nyc.jpg" width=128 height=128>
            <p align=center>nyc.jpg</p>
        </td>
        <td>
            <img src="../proj5/out/mask-nyc.jpg" width=128 height=128>
            <p align=center>mask</p>
        </td>
                <td>
            <img src="../proj5/out/replace-nyc.jpg" width=128 height=128>
            <p align=center>to replace</p>
        </td>
        <td>
            <img src="../proj5/out/inpainting-nyc.jpg" width=128 height=128>
            <p align=center>inpainted</p>
        </td>
    </tr>
</table>
</center>

### sh
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/sh.jpeg" width=128 height=128>
            <p align=center>sh.jpg</p>
        </td>
        <td>
            <img src="../proj5/out/mask-sh.jpg" width=128 height=128>
            <p align=center>mask</p>
        </td>
                <td>
            <img src="../proj5/out/replace-sh.jpg" width=128 height=128>
            <p align=center>to replace</p>
        </td>
        <td>
            <img src="../proj5/out/inpainting-sh.jpg" width=128 height=128>
            <p align=center>inpainted</p>
        </td>
    </tr>
</table>
</center>

## 1.7.3 Text Conditional Image to Image Translation
We are going to run the image translation again, but we'll replace the generic embedding `"a high quality photo"` into a specific prompt. The generated models will look more like either the prompt or the original image passed into the model depending on how noisy the initial forwarding process is.

### `"a rocket ship"` $\longrightarrow$ campanile.jpg
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/text-translate-campanile-1.jpg" width=128 height=128>
            <p align=center>i_start=0</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-campanile-3.jpg" width=128 height=128>
            <p align=center>i_start=3</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-campanile-5.jpg" width=128 height=128>
            <p align=center>i_start=5</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-campanile-7.jpg" width=128 height=128>
            <p align=center>i_start=7</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-campanile-10.jpg" width=128 height=128>
            <p align=center>i_start=10</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-campanile-20.jpg" width=128 height=128>
            <p align=center>i_start=20</p>
        </td>
        <td>
            <img src="../proj5/out/campanile.jpg" width=128 height=128>
            <p align=center>campanile.jpg</p>
        </td>
    </tr>
</table>
</center>

### `"a lithograph of waterfalls"` $\longrightarrow$ nyc.jpg
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/text-translate-nyc-1.jpg" width=128 height=128>
            <p align=center>i_start=0</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-nyc-3.jpg" width=128 height=128>
            <p align=center>i_start=3</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-nyc-5.jpg" width=128 height=128>
            <p align=center>i_start=5</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-nyc-7.jpg" width=128 height=128>
            <p align=center>i_start=7</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-nyc-10.jpg" width=128 height=128>
            <p align=center>i_start=10</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-nyc-20.jpg" width=128 height=128>
            <p align=center>i_start=20</p>
        </td>
        <td>
            <img src="../proj5/out/nyc.jpg" width=128 height=128>
            <p align=center>nyc.jpg</p>
        </td>
    </tr>
</table>
</center>

### `"an oil painting of a snowy mountain village"` $\longrightarrow$ sf.jpg
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/text-translate-sf-1.jpg" width=128 height=128>
            <p align=center>i_start=0</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-sf-3.jpg" width=128 height=128>
            <p align=center>i_start=3</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-sf-5.jpg" width=128 height=128>
            <p align=center>i_start=5</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-sf-7.jpg" width=128 height=128>
            <p align=center>i_start=7</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-sf-10.jpg" width=128 height=128>
            <p align=center>i_start=10</p>
        </td>
        <td>
            <img src="../proj5/out/text-translate-sf-20.jpg" width=128 height=128>
            <p align=center>i_start=20</p>
        </td>
        <td>
            <img src="../proj5/out/sf.jpg" width=128 height=128>
            <p align=center>sf.jpg</p>
        </td>
    </tr>
</table>
</center>

## 1.8 Visual Anagrams
We can create optical illusions with diffusion models by using the [Visual Anagrams](https://dangeng.github.io/visual_anagrams/) algorithm presented by this paper. Basically, we take two images and generate their CFG noise and then combine the noise two noises. However, one of the images must be flipped and then flipped again to generate an optical illusion that can be seen when the image is flipped. For this project, I just flipped along the x-axis (index 2 of the tensor) using `torch.flip`.

$$
\epsilon_1 = \text{UNet}(x_t, t, p_1)
$$

$$
\epsilon_2 = \text{flip}(\text{UNet}(\text{flip}(x_t), t, p_2))
$$

$$
\epsilon = (\epsilon_1 + \epsilon_2) / 2
$$

Here are some examples:

<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/visual-anagram-oldman.jpg" width=128 height=128>
            <p align=center>old man</p>
        </td>
        <td>
            <img src="../proj5/out/visual-anagram-campfire.jpg" width=128 height=128>
            <p align=center>campfire</p>
        </td>
    </tr>
</table>
</center>

<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/visual-anagram-rocket.jpg" width=128 height=128>
            <p align=center>rocket ship</p>
        </td>
        <td>
            <img src="../proj5/out/visual-anagram-snow.jpg" width=128 height=128>
            <p align=center>snowy mountain village</p>
        </td>
    </tr>
</table>
<p align=center>you can see the village roofs</p>
</center>

<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/visual-anagram-dog.jpg" width=128 height=128>
            <p align=center>dog</p>
        </td>
        <td>
            <img src="../proj5/out/visual-anagram-waterfalls.jpg" width=128 height=128>
            <p align=center>waterfall</p>
        </td>
    </tr>
</table>
</center>

## 1.9 Hybrid Images
We can also create hybrid images by calculating the CFG noise of the two images and then combining the low frequency of one image with the high frequency of another image as demonstrated with this paper on [Factorized Diffusion](https://arxiv.org/abs/2404.11615).

$$
\epsilon_1 = \text{UNet}(x_t, t, p_1)
$$

$$
\epsilon_2 = \text{UNet}(x_t, t, p_2)
$$

$$
\epsilon = f_\text{lowpass}(\epsilon_1) + f_\text{highpass}(\epsilon_2)
$$

Here are some examples:
<center>
<table>
    <tr>
        <td>
            <img src="../proj5/out/hybrid-skull-waterfall.jpg" width=128 height=128>
            <p align=center>skull + waterfall</p>
        </td>
        <td>
            <img src="../proj5/out/hybrid-yin-yang-flowers.jpg" width=128 height=128>
            <p align=center>yin and yang + flowers</p>
        </td>
        <td>
            <img src="../proj5/out/hybrid-panda-sunset.jpg" width=128 height=128>
            <p align=center>panda + sunset</p>
        </td>
    </tr>
</table>
</center>

### proj5a reflection
I really enjoyed this project as it was my first time using a diffusion model. It was fun creating hybrid and anagram images. I learned a lot about how diffusion models work and hopefully I could do a deeper dive into diffusion models with 5b.

# Proj5b:
[back to project list](../index.md)
