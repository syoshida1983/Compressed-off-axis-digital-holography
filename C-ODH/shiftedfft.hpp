/**
 * @file shiftedfft.hpp
 * @brief functions of DC centered (inverse) fast Fourier transform using Math Kernel Library for Armadillo
 * @details coefficient for energy conservation is multiplied by the IFFT result
 */

#pragma once

#include <armadillo>
#include "fft.hpp"
#include "fftshift.hpp"

/**
 * @namespace fft
 * @brief fast Fourier transform using Math Kernel Library for Armadillo
 */
namespace fft
{
	using namespace std;

	/**
	 * @brief in-place 1D fast Fourier transform (DC center)
	 * @param[in,out] f input-output
	 */
	void shiftedfft(arma::cx_vec &f)
	{
		fft::fftshift(f);
		fft::fft(f);
		fft::fftshift(f);
	}

	/**
	 * @brief 1D fast Fourier transform (DC center)
	 * @param[in] f input
	 * @return Fourier coefficients
	 */
	template < class COL, class T >
	COL shiftedfft(const arma::Col< T > &f)
	{
		arma::cx_vec x = arma::conv_to< arma::cx_vec >::from(f);
		fft::fftshift(x);
		fft::fft(x);
		fft::fftshift(x);
		return arma::conv_to< COL >::from(x);
	}

	/**
	 * @brief in-place 2D fast Fourier transform (DC center)
	 * @param[in,out] f input-output
	 */
	void shiftedfft(arma::cx_mat &f)
	{
		fft::fftshift(f);
		fft::fft(f);
		fft::fftshift(f);
	}

	/**
	 * @brief 2D fast Fourier transform (DC center)
	 * @param[in] f input
	 * @return Fourier coefficients
	 */
	template < class MAT, class T >
	MAT shiftedfft(const arma::Mat< T > &f)
	{
		arma::cx_mat x = arma::conv_to< arma::cx_mat >::from(f);
		fft::fftshift(x);
		fft::fft(x);
		fft::fftshift(x);
		return arma::conv_to< MAT >::from(x);
	}

	/**
	 * @brief in-place 3D fast Fourier transform (DC center)
	 * @param[in,out] f input-output
	 */
	void shiftedfft(arma::cx_cube &f)
	{
		fft::fftshift(f);
		fft::fft(f);
		fft::fftshift(f);
	}

	/**
	 * @brief 3D fast Fourier transform (DC center)
	 * @param[in] f input
	 * @return Fourier coefficients
	 */
	template < class CUBE, class T >
	CUBE shiftedfft(const arma::Cube< T > &f)
	{
		arma::cx_cube x = arma::conv_to< arma::cx_cube >::from(f);
		fft::fftshift(x);
		fft::fft(x);
		fft::fftshift(x);
		return arma::conv_to< CUBE >::from(x);
	}

	/**
	 * @brief in-place 1D inverse fast Fourier transform (DC center)
	 * @param[in,out] f input-output
	 */
	void shiftedifft(arma::cx_vec &f)
	{
		fft::fftshift(f);
		fft::ifft(f);
		fft::fftshift(f);
	}

	/**
	 * @brief 1D inverse fast Fourier transform (DC center)
	 * @param[in] F Fourier coefficients
	 * @return output
	 */
	template < class COL, class T >
	COL shiftedifft(const arma::Col< T > &F)
	{
		arma::cx_vec x = arma::conv_to< arma::cx_vec >::from(F);
		fft::fftshift(x);
		fft::ifft(x);
		fft::fftshift(x);
		return arma::conv_to< COL >::from(x);
	}

	/**
	 * @brief in-place 2D inverse fast Fourier transform (DC center)
	 * @param[in,out] f input-output
	 */
	void shiftedifft(arma::cx_mat &f)
	{
		fft::fftshift(f);
		fft::ifft(f);
		fft::fftshift(f);
	}

	/**
	 * @brief 2D inverse fast Fourier transform (DC center)
	 * @param[in] F Fourier coefficients
	 * @return output
	 */
	template< class MAT, class T >
	MAT shiftedifft(const arma::Mat< T > &F)
	{
		arma::cx_mat x = arma::conv_to< arma::cx_mat >::from(F);
		fft::fftshift(x);
		fft::ifft(x);
		fft::fftshift(x);
		return arma::conv_to< MAT >::from(x);
	}

	/**
	 * @brief in-place 3D inverse fast Fourier transform (DC center)
	 * @param[in,out] f input-output
	 */
	void shiftedifft(arma::cx_cube &f)
	{
		fft::fftshift(f);
		fft::ifft(f);
		fft::fftshift(f);
	}

	/**
	 * @brief 3D inverse fast Fourier transform (DC center)
	 * @param[in] F Fourier coefficients
	 * @return output
	 */
	template < class CUBE, class T >
	CUBE shiftedifft(const arma::Cube< T > &F)
	{
		arma::cx_cube x = arma::conv_to< arma::cx_cube >::from(F);
		fft::fftshift(x);
		fft::ifft(x);
		fft::fftshift(x);
		return arma::conv_to< CUBE >::from(x);
	}
}
