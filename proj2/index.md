---
author: "Arnold Cai"
date: "2024.9.23"
---

# proj2: filters and frequencies

## task 1: fun with filters

### 1.1: finite difference operator

I first found partial derivative matrices by convolving `Dx = np.array([[1, -1]])` and `Dy = np.array([[1], [-1]]` with `cameraman.png` as a gray scale image matrix. I used `scipy.signal.convolve2d` with `mode="same` to keep the dimensionality of the matrix after convolutions. Afterwards, I created a gradient magnitude matrix by using the two partial derivative matrices derived earlier. `g_m = np.sqrt(dx ** 2 + dy ** 2)`.

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/task1.1-dx.jpg" width=200>
            <p align="middle">dx</p>
        </td>
        <td>
            <img src="../proj2/out/task1.1-dy.jpg" width=200>
            <p align="middle">dy</p>
        </td>
        <td>
            <img src="../proj2/out/task1.1-gm.jpg" width=200>
            <p align="middle">gradient magnitude</p>
        </td>
        <td>
            <img src="../proj2/out/task1.1-bin-gm.jpg" width=200>
            <p align="middle">binarized, thresh=0.25</p>
        </td>
    </tr>
</table>
</div>

### 1.2: derivative of gaussian filter (DoG)

I applied a gaussian blur first and then found the partial derivative matrices of the blurred image to see if it helps with edge detection. The gaussian kernel was created using `cv2.getGaussianKernel`, and the 2d gaussian filter was creating doing the outer product of the gaussian kernel on itself. The kernel size `k` was `10 x 10` and the sigma was `(k-1) / 6` since ["the length of for the 99th percentile of gaussian pdf is `6 * sigma`"](https://stackoverflow.com/a/62002971).

<div align="middle">
    <img src="../proj2/out/task1.2-blurred.jpg">
    <p>gaussian blurred cameraman.png</p>
</div>

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/task1.2-dx.jpg" width=200>
            <p align="middle">dx of blurred image</p>
        </td>
        <td>
            <img src="../proj2/out/task1.2-dy.jpg" width=200>
            <p align="middle">dy of blurred image</p>
        </td>
        <td>
            <img src="../proj2/out/task1.2-gm.jpg" width=200>
            <p align="middle">gradient magnitude</p>
        </td>
        <td>
            <img src="../proj2/out/task1.2-bin-gm.jpg" width=200>
            <p align="middle">binarized, thresh=0.05</p>
        </td>
    </tr>
</table>
</div>

Second method I tried was to first blur the derivative matrices by convolving the gaussing filter with the finite difference matrices. Afterwards, I convolved the newly transformed gaussian filters with the original image to find the gradient magnitude.

<div align="middle">
    <table>
        <tr>
            <td>
                <img src="../proj2/out/task1.2-dog-dogx.jpg" width=200>
                <p align="middle">dx of gaussian</p>
            </td>
            <td>
                <img src="../proj2/out/task1.2-dog-dogy.jpg" width=200>
                <p align="middle">dy of gaussian</p>
            </td>
        </tr>
    </table>
</div>

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/task1.2-dog-dx.jpg" width=200>
            <p align="middle">dx of blurred image</p>
        </td>
        <td>
            <img src="../proj2/out/task1.2-dog-dy.jpg" width=200>
            <p align="middle">dy of blurred image</p>
        </td>
        <td>
            <img src="../proj2/out/task1.2-dog-gm.jpg" width=200>
            <p align="middle">gradient magnitude</p>
        </td>
        <td>
            <img src="../proj2/out/task1.2-dog-bin-gm.jpg" width=200>
            <p align="middle">binarized, thresh=0.05</p>
        </td>
    </tr>
</table>
</div>

Both methods work well and the output resutls look basically the same. There might be slightly some more noise in the first one compared to the second one, but it is only noticable when gone a thorough examination of both images.

## task 2: fun with frequencies

### 2.1:  sharpening
Steps to sharpening an image:
1. Extract low frequencies of image via low pass filter. I used gaussian blur.
2. Extract high frequenceis of image via `image - low`.
3. Add high frequencies multipled by alpha back to image via `image + alpha * high`.

#### taj.jpg with alpha=1

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/task2.1-taj-1-low.jpg" width=200>
            <p align="middle">taj.jpg</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-taj-1-low.jpg" width=200>
            <p align="middle">low taj.jpg</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-taj-1-high.jpg" width=200>
            <p align="middle">high taj.jpg</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-taj-1-final.jpg" width=200>
            <p align="middle">sharpened taj.jpg, alpha=1</p>
        </td>
    </tr>
</table>
</div>

#### side note
> I used `cv2` operations since they automatically deal with out of range values. I tried using `np.clip` after doing np matrix operations before but `cv2` operations do a much better job.

#### taj.jpg
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/taj.jpg" width=200>
            <p align="middle">alpha=0</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-taj-1-final.jpg" width=200>
            <p align="middle">alpha=1</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-taj-2-final.jpg" width=200>
            <p align="middle">alpha=2</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-taj-5-final.jpg" width=200>
            <p align="middle">alpha=5</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-taj-20-final.jpg" width=200>
            <p align="middle">alpha=20</p>
        </td>
    </tr>
</table>
</div>

#### mlord.png
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/mlord.png" width=200>
            <p align="middle">alpha=0</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-mlord-1-final.jpg" width=200>
            <p align="middle">alpha=1</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-mlord-2-final.jpg" width=200>
            <p align="middle">alpha=2</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-mlord-5-final.jpg" width=200>
            <p align="middle">alpha=5</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-mlord-20-final.jpg" width=200>
            <p align="middle">alpha=20</p>
        </td>
    </tr>
</table>
</div>

#### nostudy.png
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/nostudy.png" width=200>
            <p align="middle">alpha=0</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-nostudy-1-final.jpg" width=200>
            <p align="middle">alpha=1</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-nostudy-2-final.jpg" width=200>
            <p align="middle">alpha=2</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-nostudy-5-final.jpg" width=200>
            <p align="middle">alpha=5</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-nostudy-20-final.jpg" width=200>
            <p align="middle">alpha=20</p>
        </td>
    </tr>
</table>
</div>

I also tried "resharpening" an image by blurring an already sharp image and then sharpening it again.

#### nosleep.jpg

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/task2.1-nosleep-first-2-final.jpg" width=200>
            <p align="middle">initial sharpened image, alpha=2</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-nosleep-second-2-low.jpg" width=200>
            <p align="middle">blur of initial image</p>
        </td>
        <td>
            <img src="../proj2/out/task2.1-nosleep-second-2-final.jpg" width=200>
            <p align="middle">sharpen, alpha=2</p>
        </td>
    </tr>
</table>
</div>

>The resharpened image has more clear edges but has weird artifacts, presumably from creating previously nonexistant edges into edges. e.g. the face shading now has a bunch of weird cracks now.

### 2.2: hybrid images
To make make some hybrid images, align an the two images and then sum up one image's low frequencies and the other's high frequencies.

#### derek and nutmeg
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/DerekPicture.jpg" width=200>
            <p align="middle">derek</p>
        </td>
        <td>
            <img src="../proj2/out/nutmeg.jpg" width=200>
            <p align="middle">nutmeg</p>
        </td>
        <td>
            <img src="../proj2/out/task2.2-cat-human-hybrid.jpg" width=200>
            <p align="middle">a furry</p>
        </td>
    </tr>
</table>
</div>

#### chimera
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/fmanina.png" width=200>
            <p align="middle">nina from full metal alchemist</p>
        </td>
        <td>
            <img src="../proj2/out/fmadog.png" width=200>
            <p align="middle">nina's dog</p>
        </td>
        <td>
            <img src="../proj2/out/task2.2-fma-hybrid.jpg" width=200>
            <p align="middle">...</p>
        </td>
    </tr>
</table>
</div>

#### gogeta
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/goku.png" width=200>
            <p align="middle">goku</p>
        </td>
        <td>
            <img src="../proj2/out/vegeta.png" width=200>
            <p align="middle">vegeta</p>
        </td>
        <td>
            <img src="../proj2/out/task2.2-gogeta.jpg" width=200>
            <p align="middle">fusion!</p>
        </td>
    </tr>
</table>
</div>

### frequency analysis

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/task2.2-goku-fft.jpg" width=200>
            <p align="middle">goku fft</p>
        </td>
       <td>
            <img src="../proj2/out/task2.2-goku-high-fft.jpg" width=200>
            <p align="middle">goku high freq fft</p>
        </td>
    </tr>
        <tr>
        <td>
            <img src="../proj2/out/task2.2-vegeta-fft.jpg" width=200>
            <p align="middle">vegeta fft</p>
        </td>
        <td>
            <img src="../proj2/out/task2.2-vegeta-low-fft.jpg" width=200>
            <p align="middle">vegeta low freq fft</p>
        </td>
    </tr>
    <tr>
        <td>
            <img src="../proj2/out/task2.2-gogeta-fft.jpg" width=200>
            <p align="middle">gogeta fft</p>
        </td>
    </tr>
</table>
</div>

The fft shows how the images align their frequencies an create the hybrid image of gogeta. You can tell via the white lines of frequencies from both images.

### 2.3: gaussian and laplacian stack

I did each stack to 10 layers.

#### gaussian stack of apple

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/task2.3-apple-gaus-0.jpg" width=200>
            <p align="middle">layer 0</p>
        </td>
        <td>
            <img src="../proj2/out/task2.3-apple-gaus-3.jpg" width=200>
            <p align="middle">layer 3</p>
        </td>
         <td>
            <img src="../proj2/out/task2.3-apple-gaus-6.jpg" width=200>
            <p align="middle">layer 6</p>
        </td>
        <td>
            <img src="../proj2/out/task2.3-apple-gaus-10.jpg" width=200>
            <p align="middle">layer 10</p>
        </td>
    </tr>
</table>
</div>

#### gaussian stack of orange
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/task2.3-orange-gaus-0.jpg" width=200>
            <p align="middle">layer 0</p>
        </td>
        <td>
            <img src="../proj2/out/task2.3-orange-gaus-3.jpg" width=200>
            <p align="middle">layer 3</p>
        </td>
         <td>
            <img src="../proj2/out/task2.3-orange-gaus-6.jpg" width=200>
            <p align="middle">layer 6</p>
        </td>
        <td>
            <img src="../proj2/out/task2.3-orange-gaus-10.jpg" width=200>
            <p align="middle">layer 10</p>
        </td>
    </tr>
</table>
</div>

#### laplacian stack of apple

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/task2.3-apple-lap-0.jpg" width=200>
            <p align="middle">layer 0</p>
        </td>
        <td>
            <img src="../proj2/out/task2.3-apple-lap-3.jpg" width=200>
            <p align="middle">layer 3</p>
        </td>
         <td>
            <img src="../proj2/out/task2.3-apple-lap-6.jpg" width=200>
            <p align="middle">layer 6</p>
        </td>
        <td>
            <img src="../proj2/out/task2.3-apple-lap-10.jpg" width=200>
            <p align="middle">layer 10</p>
        </td>
    </tr>
</table>
</div>

#### laplacian stack of orange
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj2/out/task2.3-orange-lap-0.jpg" width=200>
            <p align="middle">layer 0</p>
        </td>
        <td>
            <img src="../proj2/out/task2.3-orange-lap-3.jpg" width=200>
            <p align="middle">layer 3</p>
        </td>
         <td>
            <img src="../proj2/out/task2.3-orange-lap-6.jpg" width=200>
            <p align="middle">layer 6</p>
        </td>
        <td>
            <img src="../proj2/out/task2.3-orange-lap-10.jpg" width=200>
            <p align="middle">layer 10</p>
        </td>
    </tr>
</table>
</div>

### 2.4: multiresolution blending

#### oraple
please forigve me as i accidentally did 1 more layer than the paper itself.
<div align="middle">
<table>
    <tr>
        <td>
            <p align="middle">layer 0</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-oraple-0-l1-mask.jpg" width=200>
            <p align="middle">apple</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-oraple-0-l2-mask.jpg" width=200>
            <p align="middle">orange</p>
        </td>
         <td>
            <img src="../proj2/out/task2.4-oraple-0.jpg" width=200>
            <p align="middle">combined</p>
        </td>
    </tr>
    <tr>
        <td>
            <p align="middle">layer 2</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-oraple-2-l1-mask.jpg" width=200>
            <p align="middle">apple</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-oraple-2-l2-mask.jpg" width=200>
            <p align="middle">orange</p>
        </td>
         <td>
            <img src="../proj2/out/task2.4-oraple-2.jpg" width=200>
            <p align="middle">combined</p>
        </td>
    </tr>
    <tr>
        <td>
            <p align="middle">layer 4</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-oraple-4-l1-mask.jpg" width=200>
            <p align="middle">apple</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-oraple-4-l2-mask.jpg" width=200>
            <p align="middle">orange</p>
        </td>
         <td>
            <img src="../proj2/out/task2.4-oraple-4.jpg" width=200>
            <p align="middle">combined</p>
        </td>
    </tr>
    <tr>
        <td>
            <p align="middle">layer 7</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-oraple-7-l1-mask.jpg" width=200>
            <p align="middle">apple</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-oraple-7-l2-mask.jpg" width=200>
            <p align="middle">orange</p>
        </td>
         <td>
            <img src="../proj2/out/task2.4-oraple-7.jpg" width=200>
            <p align="middle">combined</p>
        </td>
    </tr>
</table>
</div>

<table>
    <tr>
        <td>
            <img src="../proj2/out/apple.jpeg" width=200>
            <p align="middle">apple</p>
        </td>
        <td>
            <img src="../proj2/out/orange.jpeg" width=200>
            <p align="middle">orange</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-oraple-mask.jpg" width=200>
            <p align="middle">mask</p>
        </td>
         <td>
            <img src="../proj2/out/task2.4-oraple-blend.jpg" width=200>
            <p align="middle">combined</p>
        </td>
    </tr>
</table>

#### oraple horizontal
<table>
    <tr>
        <td>
            <img src="../proj2/out/apple.jpeg" width=200>
            <p align="middle">apple</p>
        </td>
        <td>
            <img src="../proj2/out/orange.jpeg" width=200>
            <p align="middle">orange</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-oraple-horizontal-mask.jpg" width=200>
            <p align="middle">mask</p>
        </td>
         <td>
            <img src="../proj2/out/task2.4-oraple-blend.jpg" width=200>
            <p align="middle">combined</p>
        </td>
    </tr>
</table>

#### kirby
<table>
    <tr>
        <td>
            <img src="../proj2/out/kirby.jpg" width=200>
            <p align="middle">kirby</p>
        </td>
        <td>
            <img src="../proj2/out/kirby_blue.jpg" width=200>
            <p align="middle">kirby blue</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-kirby-mask.jpg" width=200>
            <p align="middle">mask</p>
        </td>
         <td>
            <img src="../proj2/out/task2.4-kirby-blend.jpg" width=200>
            <p align="middle">combined</p>
        </td>
    </tr>
</table>

#### gudetama breakfast (fail)
<table>
    <tr>
        <td>
            <img src="../proj2/out/gudetama.png" width=200>
            <p align="middle">gudetama</p>
        </td>
        <td>
            <img src="../proj2/out/breakfast.jpg" width=200>
            <p align="middle">breakfast</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-gudetama-mask.jpg" width=200>
            <p align="middle">mask</p>
        </td>
         <td>
            <img src="../proj2/out/task2.4-gudetama-blend.jpg" width=200>
            <p align="middle">combined</p>
        </td>
    </tr>
</table>

#### here's a cursed egg instead
<table>
    <tr>
        <td>
            <img src="../proj2/out/egg.jpeg" width=200>
            <p align="middle">gudetama</p>
        </td>
        <td>
            <img src="../proj2/out/breakfast.jpg" width=200>
            <p align="middle">breakfast</p>
        </td>
        <td>
            <img src="../proj2/out/task2.4-cursed-egg-mask.jpg" width=200>
            <p align="middle">mask</p>
        </td>
         <td>
            <img src="../proj2/out/task2.4-cursed-egg-blend.jpg" width=200>
            <p align="middle">combined</p>
        </td>
    </tr>
</table>


### reflection
pretty fun project overall. learned how frequencies worked and basically how photoshop works with masking. made some fun references to some of my favorite animes :).

[back to project list](../index.md)
