/**
 * @file fftshift.hpp
 * @brief functions for shifting the zero-frequency component to the center of the spectrum
 */

#pragma once

#include <armadillo>

/**
 * @namespace fft
 * @brief fast Fourier transform using Math Kernel Library for Armadillo
 */
namespace fft
{
	using namespace std;

	/**
	 * @brief shift the zero-frequency component to the center of the spectrum
	 * @param[in,out] u input-output
	 */
	template < class T >
	void fftshift(arma::Col< T > &u)
	{
		const auto N = u.n_elem;

		for (auto i = 0; i < N / 2; i++)
		{
			swap(u(i), u(i + N / 2));
		}
	}

	/**
	 * @brief shift the zero-frequency component to the center of the spectrum
	 * @param[in,out] u input-output
	 */
	template < class T >
	void fftshift(arma::Mat< T > &u)
	{
		const auto N1 = u.n_rows;
		const auto N2 = u.n_cols;

		for (auto j = 0; j < N2 / 2; j++)
		{
			for (auto i = 0; i < N1 / 2; i++)
			{
				swap(u(i, j), u(i + N1 / 2, j + N2 / 2));
				swap(u(i + N1 / 2, j), u(i, j + N2 / 2));
			}
		}
	}

	/**
	 * @brief shift the zero-frequency component to the center of the spectrum
	 * @param[in,out] u input-output
	 */
	template < class T >
	void fftshift(arma::Cube< T > &u)
	{
		const auto N1 = u.n_rows;
		const auto N2 = u.n_cols;
		const auto N3 = u.n_slices;

		for (auto k = 0; k < N3 / 2; k++)
		{
			for (auto j = 0; j < N2 / 2; j++)
			{
				for (auto i = 0; i < N1 / 2; i++)
				{
					swap(u(i, j, k), u(i + N1 / 2, j + N2 / 2, k + N3 / 2));
					swap(u(i + N1 / 2, j, k), u(i, j + N2 / 2, k + N3 / 2));
					swap(u(i, j + N2 / 2, k), u(i + N1 / 2, j, k + N3 / 2));
					swap(u(i, j, k + N3 / 2), u(i + N1 / 2, j + N2 / 2, k));
				}
			}
		}
	}
}
