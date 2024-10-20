---
author: "Arnold Cai"
date: "2024.10.20
---

# proj4a: Image Stitching and Mosaic

In the first part of this project, I will be attempting to stich photos taken from the same focal point but at different angles by using homographies and 2-band frequency blending.

## Photos Used

#### wallet for rectification
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/wallet.jpg" width=300>
        </td>
    </tr>
</table>
</div>

#### laptop for rectification
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/laptop.jpg" width=300>
        </td>
    </tr>
</table>
</div>


#### desk for mosaic stitching
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/desk-0.jpg" width=300>
        </td>
        <td>
            <img src="../proj4/out/desk-1.jpg" width=300>
        </td>
    </tr>
</table>
</div>


#### another messy desk for mosaic stitching
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/monitor-0.jpg" width=300>
        </td>
        <td>
            <img src="../proj4/out/monitor-1.jpg" width=300>
        </td>
    </tr>
</table>
</div>

#### bed for mosaic stitching
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/bed-0.jpg" width=300>
        </td>
        <td>
            <img src="../proj4/out/bed-1.jpg" width=300>
        </td>
        <td>
            <img src="../proj4/out/bed-2.jpg" width=300>
        </td>
    </tr>
</table>
</div>

#### safeway for mosaic stitching
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/safeway-0.jpg" width=300>
        </td>
        <td>
            <img src="../proj4/out/safeway-1.jpg" width=300>
        </td>
        <td>
            <img src="../proj4/out/safeway-2.jpg" width=300>
        </td>
    </tr>
</table>
</div>


## Recovering Homographies
We need to use a homography matrix in other to conduct a perspective warp (or transformation). A perspective warp utilizes eight degrees of freedom to warp one image to another.

The Homography matrix H is defined to be:

<div align="middle">
    <img src="../proj4/out/homography.png">
</div>

To find the the "weights" of the homography matrix, I used the formula described in this [paper](https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf). To summarize, given `src_points = (x1, y1)` and `dest_points = (x2, y2)`, I create a matrix `A` such that `a_x = [-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2]` and `a_y = [0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2]`. I then compute the SVD of matrix `A` to find `V.T`. The last column of `V.T` reshaped into a `3x3` matrix and then normalized such that the last element at index `(2, 2)` (according to 0 indexing) is `1.0` is the Homography matrix we are looking for.

> Side note: To normalize the Homography matrix with respect to the last element is just to divide each element of the column vector `V.T` by the last element of the `V.T` and then respahing `V.T` into the shape `(3, 3)`

## Rectification

To test the homography matrices, I first took some photos of rectangular objects at an angle and then warped those images to rectangles (aka rectifying them). Just to note, the `src_points` of `computeHomography` come from keypoints mapped on the angled photo while the `dest_points` of `computeHomography` is just some points that form a rectangle that is relative to the size of the image.

#### laptop.jpg
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/laptop.jpg" width=300>
            <p align="middle">laptop</p>
        </td>
        <td>
            <img src="../proj4/out/laptop-rect.jpg" width=300>
            <p align="middle">laptop rectified</p>
        </td>
    </tr>
</table>
</div>

#### wallet.jpg
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/wallet.jpg" width=300>
            <p align="middle">wallet</p>
        </td>
        <td>
            <img src="../proj4/out/wallet-rect.jpg" width=300>
            <p align="middle">wallet rectified</p>
        </td>
    </tr>
</table>
</div>

## Mosaic Stitching

I used 2-frequency blending to blend the mosaic together. This was achieved by using laplacian stacks on the images and a gaussian stack on the mask with a depth of `2`. The boundary in the mask was determined to be the middle split of region of intersections between the images.

### Caveats
Since my images were huge (around `4000 x 3000`), my gaussian kernel had to be huge for the blur between edges to be noticable. However, having a huge kernel takes a long time to compute. After some testing, a kernel with `k = 25` took around one minute to compute; a kernel with `k = 50` took around five minutes to compute; a kernel with `k = 100` took around twenty minutes to compute. I ended up going with using `k = 50` to compute my blurs since it produced a similar output as `k = 100` with a reasonable runtime.

My phone camera seems to ***auto adjust*** the lighting after taking a photo, and I do not know how to turn off this feature. Therefore, my photos that I stitched together seems to have different lighting even though they were taken in the same setting at the same time. This caused the images look a bit funky after the stitching process. I tried my best to take photos in a setting where the auto color correction wouldn't change too much in between photos, but it is still a tiny bit noticable in some photos. Maybe I can code an auto color correction algorithm such that the images are in the relatively same lighting in the future to fix this.

#### desk.jpg
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/desk-0.jpg" width=300>
            <p align="middle">left</p>
        </td>
        <td>
            <img src="../proj4/out/desk-1.jpg" width=300>
            <p align="middle">right</p>
        </td>
    </tr>
</table>
<img src="../proj4/out/desk-mosaic-1.jpg">
<p align="middle">final</p>
</div>

### [failed] desk.jpg with left, right, middle
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/desk-0.jpg" width=300>
            <p align="middle">left</p>
        </td>
        <td>
            <img src="../proj4/out/desk-1.jpg" width=300>
            <p align="middle">middle</p>
        </td>
        <td>
            <img src="../proj4/out/desk-2.jpg" width=300>
            <p align="middle">right</p>
        </td>
    </tr>
</table>
<img src="../proj4/out/desk-mosaic-2.jpg">
<p align="middle">final</p>
</div>

> I think for this bigger mosaic I messed up the when taking the photos such that the `right` photo didn't have the same `center of projection` as the `left` and `middle` photos. Either that or I messed up the corrspondence points. I think the former is the more likely culprit.

### monitor.jpg (aka messy desk part 2)

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/monitor-0.jpg" width=300>
            <p align="middle">left</p>
        </td>
        <td>
            <img src="../proj4/out/monitor-1.jpg" width=300>
            <p align="middle">right</p>
        </td>
    </tr>
</table>
<img src="../proj4/out/monitor-mosaic-1.jpg">
<p align="middle">final</p>
</div>

### [failed] bed.jpg
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/bed-0.jpg" width=300>
            <p align="middle">left</p>
        </td>
        <td>
            <img src="../proj4/out/bed-1.jpg" width=300>
            <p align="middle">right</p>
        </td>
    </tr>
</table>
<img src="../proj4/out/bed-mosaic-1.jpg">
<p align="middle">final</p>
</div>

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/bed-2.jpg" width=300>
            <p align="middle">left</p>
        </td>
        <td>
            <img src="../proj4/out/bed-1.jpg" width=300>
            <p align="middle">middle</p>
        </td>
        <td>
            <img src="../proj4/out/bed-0.jpg" width=300>
            <p align="middle">right</p>
        </td>
    </tr>
</table>
<img src="../proj4/out/bed-mosaic-2.jpg">
<p align="middle">final</p>
</div>

> Phone camera lighting auto adjustment issue as mentioned in caveats.

### safeway.jpg
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/safeway-0.jpg" width=300>
            <p align="middle">left</p>
        </td>
        <td>
            <img src="../proj4/out/safeway-1.jpg" width=300>
            <p align="middle">right</p>
        </td>
    </tr>
</table>
<img src="../proj4/out/safeway-mosaic-1.jpg">
<p align="middle">final</p>
</div>

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj4/out/safeway-0.jpg" width=300>
            <p align="middle">left</p>
        </td>
        <td>
            <img src="../proj4/out/safeway-1.jpg" width=300>
            <p align="middle">middle</p>
        </td>
        <td>
            <img src="../proj4/out/safeway-2.jpg" width=300>
            <p align="middle">right</p>
        </td>
    </tr>
</table>
<img src="../proj4/out/safeway-mosaic-2.jpg">
<p align="middle">final</p>
</div>

### 4a Reflection
I overall enjoyed taking photos and then stitching the images into mosaics. I had some trouble getting the images to have the same lighting, due to the nature of my phone camera which auto adjusts the image after it being taken. Furthermore the images taken from different angles affected how much light the camera was receiving and therefore may have affected the lighting of the photos and mad the stiching look weird and the photos with different lighting. I think I could've implemented an auto color correction or averaging algorithm between the images to solve this issue from a software standpoint, but I didn't have enough time to implement this during this project due to the heavy courseload I'm taking this semester. Maybe I will come back later, to try to implement a better stitching algorithm as well as a color correction algorithm.

<!-- # proj4b: [Auto]-Stitching and Mosaics -->

[back to project list](../index.md)
