/**
 * @file shift.hpp
 * @brief functions for circular shifting matrix elements
 */

#pragma once

#include <armadillo>
#include "shiftedfft.hpp"

/**
 * @namespace fft
 * @brief fast Fourier transform using Math Kernel Library for Armadillo
 */
namespace fft
{
	using namespace std;

	/**
	 * @brief shift matrix elements circularly along row and column
	 * @param[in,out] x    input-output
	 * @param[in]     rows the number of places by which elements are shifted along the row
	 * @param[in]     cols the number of places by which elements are shifted along the column
	 */
	template < class T >
	void shift(arma::Mat< T > &x, const long long rows, const long long cols)
	{
		x = arma::shift(arma::shift(x, rows, 0), cols, 1);
	}

	/**
	 * @brief shift matrix elements circularly along row and column in the frequency domain
	 * @param[in,out] x    input-output
	 * @param[in]     rows the number of places by which elements are shifted along the row
	 * @param[in]     cols the number of places by which elements are shifted along the column
	 */
	void frequencyshift(arma::cx_mat &x, const long long rows, const long long cols)
	{
		fft::shiftedfft(x);
		x = arma::shift(arma::shift(x, rows, 0), cols, 1);
		fft::shiftedifft(x);
	}
}
