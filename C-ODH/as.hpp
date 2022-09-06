/**
 * @file as.hpp
 * @brief functions of angular spectrum method for diffraction calculation
 */

#pragma once

#include <cmath>
#include <armadillo>
#include "shiftedfft.hpp"

/**
 * @namespace as
 * @brief angular spectrum method
 */
namespace as
{
	using namespace std;

	/**
	 * @brief in-place 1D angular spectrum method
	 * @param[in,out]  u      input-output wavefront
	 * @param[in]      lambda wavelength
	 * @param[in]      z      propagation distance
	 * @param[in]      dx     mesh size in the x-direction
	 */
	void as(
				arma::cx_vec	&u,
		const	double			lambda,
		const	double			z,
		const	double			dx)
	{
		const auto N = u.n_elem;

		fft::shiftedfft(u);

		#pragma omp parallel for
		for (auto i = 0; i < N; i++)
		{
			// spatial frequency
			const auto v = (i - N / 2.) / (N * dx);
			const complex< double > w = sqrt(complex< double >(1 / (lambda * lambda) - v * v, 0));

			u(i) *= exp(-2 * M_PI * abs(z) * imag(w)) * polar(1.0, 2 * M_PI * z * real(w));
		}

		fft::shiftedifft(u);
	}

	/**
	 * @brief 1D angular spectrum method
	 * @param[in] u      input wavefront
	 * @param[in] lambda wavelength
	 * @param[in] z      propagation distance
	 * @param[in] dx     mesh size in the x-direction
	 * @return diffracted wavefront
	 */
	template < class COL, class T >
	COL as(
		const arma::Col< T >	&u,
		const double			lambda,
		const double			z,
		const double			dx)
	{
		const auto N = u.n_elem;

		arma::cx_vec U = arma::conv_to< arma::cx_vec >::from(u);
		fft::shiftedfft(U);

		#pragma omp parallel for
		for (auto i = 0; i < N; i++)
		{
			/// spatial frequency
			const auto v = (i - N / 2.) / (N * dx);
			const complex< double > w = sqrt(complex< double >(1 / (lambda * lambda) - v * v, 0));

			U(i) *= exp(-2 * M_PI * abs(z) * imag(w)) * polar(1.0, 2 * M_PI * z * real(w));
		}

		fft::shiftedifft(U);

		return arma::conv_to< COL >::from(U);
	}

	/**
	 * @brief in-place 2D angular spectrum method
	 * @param[in,out] u      input-output wavefront
	 * @param[in]     lambda wavelength
	 * @param[in]     z      propagation distance
	 * @param[in]     dx     mesh size in the x-direction
	 * @param[in]     dy     mesh size in the y-direction
	 */
	void as(
				arma::cx_mat	&u,
		const	double			lambda,
		const	double			z,
		const	double			dx,
		const	double			dy)
	{
		const auto N1 = u.n_rows;
		const auto N2 = u.n_cols;

		fft::shiftedfft(u);

		#pragma omp parallel for
		for (auto j = 0; j < N2; j++)
		{
			for (auto i = 0; i < N1; i++)
			{
				/// spatial frequency
				const arma::vec2 v = {(i - N1 / 2.) / (N1 * dx), (j - N2 / 2.) / (N2 * dy)};
				const complex< double > w = sqrt(complex< double >(1 / (lambda * lambda) - arma::dot(v, v), 0));

				u(i, j) *= exp(-2 * M_PI * abs(z) * imag(w)) * polar(1.0, 2 * M_PI * z * real(w));
			}
		}

		fft::shiftedifft(u);
	}

	/**
	 * @brief 2D angular spectrum method
	 * @param[in] u      input wavefront
	 * @param[in] lambda wavelength
	 * @param[in] z      propagation distance
	 * @param[in] dx     mesh size in the x-direction
	 * @param[in] dy     mesh size in the y-direction
	 * @return diffracted wavefront
	 */
	template < class MAT, class T >
	MAT as(
		const arma::Mat< T >	&u,
		const double			lambda,
		const double			z,
		const double			dx,
		const double			dy)
	{
		const auto N1 = u.n_rows;
		const auto N2 = u.n_cols;

		arma::cx_mat U = arma::conv_to< arma::cx_mat >::from(u);
		fft::shiftedfft(U);

		#pragma omp parallel for
		for (auto j = 0; j < N2; j++)
		{
			for (auto i = 0; i < N1; i++)
			{
				/// spatial frequency
				const arma::vec2 v = {(i - N1 / 2.) / (N1 * dx), (j - N2 / 2.) / (N2 * dy)};
				const complex< double > w = sqrt(complex< double >(1 / (lambda * lambda) - arma::dot(v, v), 0));

				U(i, j) *= exp(-2 * M_PI * abs(z) * imag(w)) * polar(1.0, 2 * M_PI * z * real(w));
			}
		}

		fft::shiftedifft(U);

		return arma::conv_to< MAT >::from(U);
	}
}
