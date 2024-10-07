---
author: "Arnold Cai"
date: "2024.10.7
---

# proj3: it's morphin' time (face morphing)

In this project, I will be morphing one face to another and also do some experimentation with population mean faces.

## task 1: defining correspondences
For the first morph, I decided to morph George Clooney's face to Mark Zuckerberg's face and vice versa. Photos are from [Martin Schoeller's portfolio](https://martinschoeller.com).

I created a triangular mask for each photo using [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation). Both photos contain the same triangular structure as each other, so translating corresponding keypoints and triangluar regions is easier.

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj3/out/clooney.jpg" width=200>
            <p align="middle">george clooney</p>
        </td>
        <td>
            <img src="../proj3/out/clooney-keypoints.jpg" width=200>
            <p align="middle">george clooney with keypoints mask</p>
        </td>
    </tr>
</table>
</div>

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj3/out/zuck.jpg" width=200>
            <p align="middle">mark zuckerberg</p>
        </td>
        <td>
            <img src="../proj3/out/zuck-keypoints.jpg" width=200>
            <p align="middle">mark zuckerberg with keypoints mask</p>
        </td>
    </tr>
</table>
</div>

## task 2: computing "midway face"
To compute the "midway face":
1. Find the average triangulation and keypoints between the two triangular mask.
> Although the relative structure of the triangular masks are the same, the coordinates are not exactly the same and therefore we need a triangular mask that is the composed of the average of the keypoints and their respective positions.
2. Warp the images to the average triangular mask, I used inverse warping
> For each triangle in the triangular mask:
>
> 1. Find the affine transformation matrix from avg keypoints to img keypoints. This can be done by using `np.linalg.solve`, where `a` is the avg keypoints matrix and `b` is the img keypoints matrix. Make sure that each keypoint is in this format: `[x, y, 1]`, since `np.linalg.solve` solves `ax = b` for `x`. The resulting affine transformation matrix will be `A = x.T`.
> 2. Use the affine matrix to find the points that correspond to the region in the avg triangle in the img triangle by doing `A @ avg_points_matrix`.
> 3. Use nearest neighbor interpolation to find the resulting img coordinates for transformed coordinates that are in between coordinates (aka are floats and not integers)
> 4. Place the image values that are located at the img coordinates derived from `3.` into the warped image array.

3. Cross-disolve the two warped images by taking the average of the RGB values of the warped images

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj3/out/clooney.jpg" width=300>
            <p align="middle">clooney</p>
        </td>
         <td>
            <img src="../proj3/out/clooney-zuck-avg-keypoints.jpg" width=300>
            <p align="middle">clooney and mark avg keypoints</p>
        </td>
        <td>
            <img src="../proj3/out/zuck.jpg" width=300>
            <p align="middle">zuck</p>
        </td>
    </tr>
</table>
</div>

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj3/out/clooney-warped.jpg" width=300>
            <p align="middle">clooney warped</p>
        </td>
        <td>
            <img src="../proj3/out/clooney-zuck-mean.jpg" width=300>
            <p align="middle">clooney and zuck mean face</p>
        </td>
        <td>
            <img src="../proj3/out/zuck-warped.jpg" width=300>
            <p align="middle">zuck warped</p>
        </td>
    </tr>
</table>
</div>

## task 3: morph sequence
Creating a morph sequence is similar to calculating the midway face but instead of keeping a constant warp alpha and dissolve alpha at 0.50, we do a linear interoplation of the the two constants to get a smooth transistion of morphs.

To create this linear interpolation, I just used `np.linspace` to create 45 alphas that can be in between the range of `[0, 1]`, since I want to create 45 frames of in-between morph sequences. Each frame uses the same alpha for warping and cross-dissolving, which seems to work fine. However, I think a smoother transition can be done if the warp alpha was a function of dissolve alpha for some function `f`. This probably needs a bit more testing and research before much more can be said.

<div align="middle">

![clooney-zuck-morph.gif](../proj3/out/clooney-zuck.gif)

</div>


## task 4: "mean face" of population
I generated a mean face using the Danes dataset of annotated faces (30 males, 7 females).

To generate the mean face:
1. Find the average triangular mask of all the faces
2. Warp each img to the average triangular mask
3. Compute the average RGB values of all the warped images

<div align="middle">

![](../proj3/out/danes-mean-face.jpg)

danes mean face

</div>

<div align="middle">
<table>
    <tr>
         <td>
            <img src="../proj3/out/swift-to-danes.jpg" width=300>
            <p align="middle">swift warped to danes mean face</p>
        </td>
        <td>
            <img src="../proj3/out/danes-to-swift.jpg" width=300>
            <p align="middle">danes mean face warped to swift</p>
        </td>
    </tr>
</table>
</div>

#### Examples of faces in dataset warped to danes mean face
<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj3/out/07-1m-warped.jpg" width=300>
            <p align="middle">07-1m</p>
        </td>
         <td>
            <img src="../proj3/out/12-1f-warped.jpg" width=300>
            <p align="middle">12-1f</p>
        </td>
    </tr>
    <tr>
        <td>
            <img src="../proj3/out/23-1m-warped.jpg" width=300>
            <p align="middle">23-1m</p>
        </td>
         <td>
            <img src="../proj3/out/31-1m-warped.jpg" width=300>
            <p align="middle">31-1m</p>
        </td>
    </tr>
</table>
</div>

## task 5: extrapolating from the mean (danish taylor swift)
Caricatures of a face can be derived from this formula: `caricature = alpha * (average_keypoints - original_keypoints) + original_keypoints`. Here are some with examples of Taylor Swift being more or less Danish by tuning the alpha.

<div align="middle">
<table>
    <tr>
        <td>
            <img src="../proj3/out/swift-caricature--0.75.jpg" width=300>
            <p align="middle">alpha=-0.75</p>
        </td>
         <td>
            <img src="../proj3/out/swift-caricature--0.25.jpg" width=300>
            <p align="middle">alpha=-0.25</p>
        </td>
        <td>
            <img src="../proj3/out/swift-caricature-0.jpg" width=300>
            <p align="middle">alpha=0</p>
        </td>
       <td>
            <img src="../proj3/out/swift-caricature-0.25.jpg" width=300>
            <p align="middle">alpha=-0.25</p>
        </td>
        <td>
            <img src="../proj3/out/swift-caricature-0.75.jpg" width=300>
            <p align="middle">alpha=0.75</p>
        </td>
    </tr>
</table>
</div>

## bells and whistles

### reflection
pretty fun project over all and learned more the applications of affine transformations in computer vision.
[back to project list](../index.md)
