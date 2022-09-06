/**
 * @file main.cpp
 * @brief main source file
 * @details implementation of the algorithm described in the following paper https://doi.org/10.1088/2040-8986/aba940
 */

#define _USE_MATH_DEFINES
#define NOMINMAX
#define ARMA_USE_OPENMP
#define ARMA_64BIT_WORD


#include "image.hpp"
#include "window.hpp"
#include "shift.hpp"
#include "shiftedfft.hpp"
#include "as.hpp"
#include "tv.hpp"

int main()
{
	using namespace std;

	// parameters
	const string	inimg	= "USAF-1951.bmp";
	const string	outimg	= "FISTA.bmp";
	const long long	N		= 100;		///< number of iterations
	const long long	NTV		= 10;		///< number of iterations for TV regularization (FGP)
	const long long PX		= 128;		///< x-coordinate of object wave in the frequency domain
	const long long PY		= 0;		///< y-coordinate of object wave in the frequency domain
	const double	lambda	= 405e-9;	///< wavelength [m]
	const double	d		= 0.2;		///< propagation distance [m]
	const double	dx		= 3.45e-6;	///< mesh size in the x-direction [m]
	const double	dy		= 3.45e-6;	///< mesh size in the x-direction [m]
	const double	gamma	= 0.01;		///< step size
	const double	tau		= 0.8;		///< TV regularization parameter
	const double	rd		= 128;		///< radius of the window for the high pass filter
	
	// read and normalize the image
	arma::mat f = image::imread< arma::mat >(inimg).at("k") / 255;

	// preserve the original image
	const arma::mat g = f;

	const long long	NX	= f.n_cols;				///< image size in the x-direction
	const long long	NY	= f.n_rows;				///< image size in the y-direction
	const double	k0	= 2 * M_PI / lambda;	///< wavenumber
	const double	fx	= PX / (dx * NX);		///< spatial frequency of object wave in the x-direction
	const double	fy	= PY / (dy * NY);		///< spatial frequency of object wave in the y-direction

	// convert to the complex matrix
	arma::cx_mat u = as::as< arma::cx_mat >(g, lambda, d, dx, dy);

	// calculate interference fringe (hologram)
	#pragma omp parallel for
	for (auto j = 0; j < NX; j++)
	{
		for (auto i = 0; i < NY; i++)
		{
			const arma::vec2 r({ j * dx, i * dy });
			const arma::vec2 k({ k0 * lambda * fx, k0 * lambda * fy });
			f(i, j) = norm(u(i, j) + polar(1., arma::dot(k, r)));
		}
	}
	image::imwrite("hologram.bmp", f);

	// FFT of the hologram
	u = arma::conv_to< arma::cx_mat >::from(f);
	fft::shiftedfft(u);

	// high pass filter
	arma::mat h = arma::ones< arma::mat >(NY, NX);
	fft::window::hann(h, NX / 2, NY / 2, rd);
	u = u % (arma::ones< arma::mat >(NY, NX) - h);

	// IFFT
	fft::shiftedifft(u);

	// extract the real part
	const arma::mat a = arma::real(u);

	// working spaces
	arma::cx_mat w = arma::zeros< arma::cx_mat >(NY, NX);
	arma::cx_mat x = arma::zeros< arma::cx_mat >(NY, NX);
	arma::cx_mat y = arma::zeros< arma::cx_mat >(NY, NX);
	arma::cx_mat z = arma::zeros< arma::cx_mat >(NY, NX);

	// working variables for FISTA
	double r = 1;
	double s = 1;

	// log file of mean squared error
	fstream fs("MSE.csv", ios::out);
	cout << "interation\tMSE" << endl;
	fs << "interation\tMSE" << endl;

	// FISTA
	for (auto i = 0; i < N; i++)
	{
		u.fill(0);

		// calculate grad f
		u = as::as< arma::cx_mat >(z, lambda, d, dx, dy);
		fft::frequencyshift(u, -PY, -PX);
		u = arma::conv_to< arma::cx_mat >::from(2 * arma::real(u) - a);
		fft::frequencyshift(u, PY, PX);
		z = as::as< arma::cx_mat >(u, lambda, -d, dx, dy);
		x = y - 2 * gamma * z;

		// TV regularization
		x = cs::tv::fgp< arma::cx_mat >(x, gamma * tau, NTV);

		// acceleration
		s = (1 + sqrt(1 + 4 * r * r)) / 2;
		z = x + ((r - 1) / s) * (x - w);

		// update
		r = s;
		w = x;
		y = z;

		// evaluate mean squared error
		f = arma::abs(x);
		f *= arma::mean(arma::vectorise(g)) / arma::mean(arma::vectorise(f));
		cout << i << "\t" << pow(arma::norm(arma::vectorise(g) - arma::vectorise(f)), 2) / g.n_elem << endl;
		fs << i << "," << pow(arma::norm(arma::vectorise(g) - arma::vectorise(f)), 2) / g.n_elem << endl;
	}

	fs.close();

	// output image
	f = arma::abs(x);
	image::imwrite(outimg, f);

	return 0;
}
