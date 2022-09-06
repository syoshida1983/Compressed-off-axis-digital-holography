# Compressed off-axis digital holography

C-ODH is a simulation program that implements the algorithm of the following paper.

> [Shuhei Yoshida, “Compressed off-axis digital holography,” J. Opt. 22 (9), 095703 (2020).](https://doi.org/10.1088/2040-8986/aba940)

Based on a compressed sensing framework, C-ODH can reconstruct object waves at high resolution from interferogram images.

## Build and Run
C-ODH is developed with [Visual Studio 2022](https://visualstudio.microsoft.com/free-developer-offers/) on Windows. [Intel Mathkernel Library (MKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html), [boost](https://www.boost.org/), [OpenCV](https://opencv.org/), and [Armadillo](http://arma.sourceforge.net/) are also used. To build C-ODH, install Visual Studio 2022 and MKL, then install boost, OpenCV, and Armadillo with [vcpkg](https://github.com/microsoft/vcpkg). After installing vcpkg according to the instructions in the above link, install the library and integrate it with Visual Studio using the following command.

```console
> vcpkg install boost:x64-windows opencv:x64-windows armadillo:x64-windows
> vcpkg integrate install
```

To build the program, open `C-ODH.sln` in Visual Studio and type `Ctrl+F5`. Numerical parameters are written at the beginning of the main function.

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
