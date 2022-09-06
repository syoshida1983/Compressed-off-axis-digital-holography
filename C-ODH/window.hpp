/**
 * @file window.hpp
 * @brief window functions
 */

#pragma once

#include <cmath>
#include <boost/math/special_functions.hpp>
#include <boost/numeric/interval.hpp>
#include <armadillo>

/**
 * @namespace fft
 * @brief fast Fourier transform using Math Kernel Library for Armadillo
 */
namespace fft
{
	using namespace std;

	/**
	 * @namespace window
	 * @brief window functions
	 */
	namespace window
	{
		using boost::numeric::interval;

		/**
		 * @brief rectangular window
		 * @param[in,out] u  input-output
		 * @param[in]     x0 x-coordinate the center of the window
		 * @param[in]     y0 y-coordinate the center of the window
		 * @param[in]     wy window size in the x-direction
		 * @param[in]     wy window size in the y-direction
		 */
		template < class T >
		void rectangular(
					arma::Mat< T >& u,
			const	double			x0,
			const	double			y0,
			const	double			wx,
			const	double			wy)
		{
			const auto N1 = u.n_rows;
			const auto N2 = u.n_cols;

			const interval< double > xw(-wx / 2, wx / 2);
			const interval< double > yw(-wy / 2, wy / 2);

			#pragma omp parallel for
			for (auto j = 0; j < N2; j++)
			{
				for (auto i = 0; i < N1; i++)
				{
					const double x = j - x0;
					const double y = i - y0;

					if (!in(x, xw) || !in(y, yw))
					{
						u(i, j) = 0;
					}
				}
			}
		}

		/**
		 * @brief circular window
		 * @param[in,out] u  input-output
		 * @param[in]     x0 central x-coordinate of the window
		 * @param[in]     y0 central y-coordinate of the window
		 * @param[in]     r0 radius of the window
		 */
		template < class T >
		void circle(
					arma::Mat< T >& u,
			const	double			x0,
			const	double			y0,
			const	double			r0)
		{
			const auto N1 = u.n_rows;
			const auto N2 = u.n_cols;

			#pragma omp parallel for
			for (auto j = 0; j < N2; j++)
			{
				for (auto i = 0; i < N1; i++)
				{
					const double x = j - x0;
					const double y = i - y0;
					const double r = sqrt(x * x + y * y);

					if (r > r0)
					{
						u(i, j) = 0;
					}
				}
			}
		}

		/**
		 * @brief Gaussian window
		 * @param[in,out] u  input-output
		 * @param[in]     x0 central x-coordinate of the window
		 * @param[in]     y0 central y-coordinate of the window
		 * @param[in]     sd standard deviation
		 */
		template < class T >
		void gaussian(
					arma::Mat< T >& u,
			const	double			x0,
			const	double			y0,
			const	double			sd)
		{
			const auto N1 = u.n_rows;
			const auto N2 = u.n_cols;

			#pragma omp parallel for
			for (auto j = 0; j < N2; j++)
			{
				for (auto i = 0; i < N1; i++)
				{
					const double x = j - x0;
					const double y = i - y0;
					const double r = sqrt(x * x + y * y);

					u(i, j) *= exp(-r * r / (2 * sd * sd));
				}
			}
		}

		/**
		 * @brief Hann window
		 * @param[in,out] u  input-output
		 * @param[in]     x0 central x-coordinate of the window
		 * @param[in]     y0 central y-coordinate of the window
		 * @param[in]     r0 radius of the window
		 */
		template < class T >
		void hann(
					arma::Mat< T >& u,
			const	double			x0,
			const	double			y0,
			const	double			r0)
		{
			const auto N1 = u.n_rows;
			const auto N2 = u.n_cols;

			#pragma omp parallel for
			for (auto j = 0; j < N2; j++)
			{
				for (auto i = 0; i < N1; i++)
				{
					const double x = j - x0;
					const double y = i - y0;
					const double r = sqrt(x * x + y * y);

					if (r > r0)
					{
						u(i, j) = 0;
					}
					else
					{
						u(i, j) *= 0.5 - 0.5 * cos(M_PI * (r / r0 - 1));
					}
				}
			}
		}

		/**
		 * @brief Hamming window
		 * @param[in,out] u  input-output
		 * @param[in]     x0 central x-coordinate of the window
		 * @param[in]     y0 central y-coordinate of the window
		 * @param[in]     r0 radius of the window
		 */
		template < class T >
		void hamming(
					arma::Mat< T >& u,
			const	double			x0,
			const	double			y0,
			const	double			r0)
		{
			const auto N1 = u.n_rows;
			const auto N2 = u.n_cols;

			#pragma omp parallel for
			for (auto j = 0; j < N2; j++)
			{
				for (auto i = 0; i < N1; i++)
				{
					const double x = j - x0;
					const double y = i - y0;
					const double r = sqrt(x * x + y * y);

					if (r > r0)
					{
						u(i, j) = 0;
					}
					else
					{
						u(i, j) *= 0.54 - 0.46 * cos(M_PI * (r / r0 - 1));
					}
				}
			}
		}

		/**
		 * @brief Blackman window
		 * @param[in,out] u  input-output
		 * @param[in]     x0 central x-coordinate of the window
		 * @param[in]     y0 central y-coordinate of the window
		 * @param[in]     r0 radius of the window
		 */
		template < class T >
		void blackman(
					arma::Mat< T >& u,
			const	double			x0,
			const	double			y0,
			const	double			r0)
		{
			const auto N1 = u.n_rows;
			const auto N2 = u.n_cols;

			#pragma omp parallel for
			for (auto j = 0; j < N2; j++)
			{
				for (auto i = 0; i < N1; i++)
				{
					const double x = j - x0;
					const double y = i - y0;
					const double r = sqrt(x * x + y * y);

					if (r > r0)
					{
						u(i, j) = 0;
					}
					else
					{
						u(i, j) *= 0.42 - 0.5 * cos(M_PI * (r / r0 - 1)) + 0.08 * cos(2 * M_PI * (r / r0 - 1));
					}
				}
			}
		}

		/**
		 * @brief Kaiser window
		 * @param[in,out] u  input-output
		 * @param[in]     x0 central x-coordinate of the window
		 * @param[in]     y0 central y-coordinate of the window
		 * @param[in]     r0 radius of the window
		 * @param[in]     a  parameter
		 */
		template < class T >
		void kaiser(
					arma::Mat< T >& u,
			const	double			x0,
			const	double			y0,
			const	double			r0,
			const	double			a)
		{
			const auto N1 = u.n_rows;
			const auto N2 = u.n_cols;

			const double D = boost::math::cyl_bessel_i(0, M_PI * a);

			#pragma omp parallel for
			for (auto j = 0; j < N2; j++)
			{
				for (auto i = 0; i < N1; i++)
				{
					const double x = j - x0;
					const double y = i - y0;
					const double r = sqrt(x * x + y * y);

					if (r > r0)
					{
						u(i, j) = 0;
					}
					else
					{
						u(i, j) *= boost::math::cyl_bessel_i(0, M_PI * a * sqrt(1 - (r / r0) * (r / r0))) / D;
					}
				}
			}
		}
	}
}
