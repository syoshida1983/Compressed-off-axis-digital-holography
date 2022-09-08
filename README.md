# Compressed off-axis digital holography

C-ODH is a simulation program that implements the algorithm of the following paper.

> [Shuhei Yoshida, “Compressed off-axis digital holography,” J. Opt. 22 (9), 095703 (2020).](https://doi.org/10.1088/2040-8986/aba940)

Based on a compressed sensing framework, C-ODH can reconstruct object waves at high resolution from interferogram images.

## Build and Run
C-ODH is developed with [Visual Studio 2022](https://visualstudio.microsoft.com/free-developer-offers/) on Windows. [Intel Math Kernel Library (MKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html), [boost](https://www.boost.org/), [OpenCV](https://opencv.org/), and [Armadillo](http://arma.sourceforge.net/) are also used. To build C-ODH, install Visual Studio 2022 and MKL, then install boost, OpenCV, and Armadillo with [vcpkg](https://github.com/microsoft/vcpkg). After installing vcpkg according to the instructions in the above link, install the libraries and integrate them with Visual Studio using the following command.

```console
> vcpkg install boost:x64-windows opencv:x64-windows armadillo:x64-windows
> vcpkg integrate install
```

To build and run the program, open `C-ODH.sln` in Visual Studio and type `Ctrl+F5`. Numerical parameters are written at the beginning of the main function.

```cpp:main.cpp
const string	inimg	= "USAF-1951.bmp";
const string	outimg	= "FISTA.bmp";
const long long	N	= 100;		///< number of iterations
const long long	NTV	= 10;		///< number of iterations for TV regularization (FGP)
const long long PX	= 128;		///< x-coordinate of object wave in the frequency domain
const long long PY	= 0;		///< y-coordinate of object wave in the frequency domain
const double	lambda	= 405e-9;	///< wavelength [m]
const double	d	= 0.2;		///< propagation distance [m]
const double	dx	= 3.45e-6;	///< mesh size in the x-direction [m]
const double	dy	= 3.45e-6;	///< mesh size in the x-direction [m]
const double	gamma	= 0.01;		///< step size
const double	tau	= 0.8;		///< TV regularization parameter
const double	rd	= 128;		///< radius of the window for the high pass filter
```

## Numerical Model
![Numerical Model](https://github.com/syoshida1983/C-ODH/blob/images/model.jpg)

The observation model is shown in the figure above. The object wave $\mathbf{x}$ propagates to the sensor surface and interferes with the reference wave $\mathbf{r}$. This process is represented by the following linear transformation.

$$\mathbf{y}=\mathbf{Ax}\equiv 2\Re(\mathbf{F}^{*}\mathbf{P}^{\intercal}\mathbf{HFx}),$$

where $\mathbf{y}$ is the hologram with the non-diffracted wave component removed, $\mathbf{F}$ is the two-dimensional Fourier transform, $\mathbf{H}$ is the transfer function representing the propagation in free space, and $\mathbf{P}$ is an orthogonal matrix representing the frequency shift due to the superposition of the reference wave. The reconstruction of the object wave x can be formulated as the following minimization problem.

$$\min_{\mathbf{x}}\left\\{f(\mathbf{x})+g(\mathbf{x})\right\\}\equiv\min_{\mathbf{x}}\left\\{\frac{1}{2}\\|\mathbf{Ax}-\mathbf{y}\\|_{2}^{2}+\tau\mathrm{TV}(\mathbf{x})\right\\},$$

where $\\|\cdot\\|\_{2}$ is the $L_{2}$ norm, $\mathrm{TV}$ is the total variation, and $\tau$ is the regularization parameter. The above minimization problem is solved numerically using [FISTA](https://doi.org/10.1137/080716542) with initial values $\mathbf{x}=\mathbf{z}=\mathbf{0}$, $s=1$ as follows.

$$\mathbf{x}^{(n+1)}=\mathrm{prox}\_{\gamma g}\left[\mathbf{z}^{(n)}-\gamma\nabla f\left(\mathbf{z}^{(n)}\right)\right]=\mathrm{prox}\_{\gamma\tau\mathrm{TV}}\left[\mathbf{z}^{(n)}-\gamma\nabla f\left(\mathbf{z}^{(n)}\right)\right],$$

$$\nabla f(\mathbf{x})=\mathbf A^* (\mathbf{Ax}-\mathbf{y})=2\mathbf F^* \mathbf H^* \mathbf{PF}\left[2\Re\left(\mathbf F^* \mathbf P^{\intercal} \mathbf{HFx}\right)-\mathbf{y}\right],$$

$$s^{(n+1)}=\frac{1+\sqrt{1+4\left(s^{(n)}\right)^{2}}}{2},$$

$$\mathbf{z}^{(n+1)}=\mathbf{x}^{(n+1)}+\frac{s^{(n)}-1}{s^{(n+1)}}\cdot\left(\mathbf{x}^{(n+1)}-\mathbf{x}^{(n)}\right),$$

where $\gamma$ is the step size and $\mathrm{prox}\_{\gamma g}$ is the proxima operator of the function $g$ scaled by $\gamma$. [FGP](https://doi.org/10.1109/TIP.2009.2028250) is used to calculate $\mathrm{prox}\_{\gamma g}$.
