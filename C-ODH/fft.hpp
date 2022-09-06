/**
 * @file fft.hpp
 * @brief functions of (inverse) fast Fourier transform using Math Kernel Library for Armadillo
 * @details coefficient for energy conservation is multiplied by the IFFT result
 */

#pragma once

#include <mkl.h>
#include <armadillo>

/**
 * @namespace fft
 * @brief fast Fourier transform using Math Kernel Library for Armadillo
 */
namespace fft
{
	using namespace std;

	/**
	 * @brief in-place 1D fast Fourier transform
	 * @param[in,out] f input-output
	 */
	void fft(arma::cx_vec &f)
	{
		const long N = static_cast<long>(f.n_elem);

		DFTI_DESCRIPTOR_HANDLE desc_handle;
		long status;
		status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, N);
		status = DftiCommitDescriptor(desc_handle);
		status = DftiComputeForward(desc_handle, f.memptr());
		status = DftiFreeDescriptor(&desc_handle);
	}

	/**
	 * @brief 1D fast Fourier transform
	 * @param[in] f input
	 * @return Fourier coefficients
	 */
	template < class COL, class T >
	COL fft(const arma::Col< T > &f)
	{
		const long N = static_cast<long>(f.n_elem);

		arma::cx_vec x = arma::conv_to< arma::cx_vec >::from(f);

		DFTI_DESCRIPTOR_HANDLE desc_handle;
		long status;
		status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, N);
		status = DftiCommitDescriptor(desc_handle);
		status = DftiComputeForward(desc_handle, x.memptr());
		status = DftiFreeDescriptor(&desc_handle);

		return arma::conv_to< COL >::from(x);
	}

	/**
	 * @brief in-place 2D fast Fourier transform
	 * @param[in,out] f input-output
	 */
	void fft(arma::cx_mat &f)
	{
		const long N1 = static_cast<long>(f.n_rows);
		const long N2 = static_cast<long>(f.n_cols);
		const long l[] = {N2, N1};

		DFTI_DESCRIPTOR_HANDLE desc_handle;
		long status;
		status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 2, l);
		status = DftiCommitDescriptor(desc_handle);
		status = DftiComputeForward(desc_handle, f.memptr());
		status = DftiFreeDescriptor(&desc_handle);
	}

	/**
	 * @brief 2D fast Fourier transform
	 * @param[in] f input
	 * @return Fourier coefficients
	 */
	template < class MAT, class T >
	MAT fft(const arma::Mat< T > &f)
	{
		const long N1 = static_cast<long>(f.n_rows);
		const long N2 = static_cast<long>(f.n_cols);
		const long l[] = {N2, N1};

		arma::cx_mat x = arma::conv_to< arma::cx_mat >::from(f);

		DFTI_DESCRIPTOR_HANDLE desc_handle;
		long status;
		status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 2, l);
		status = DftiCommitDescriptor(desc_handle);
		status = DftiComputeForward(desc_handle, x.memptr());
		status = DftiFreeDescriptor(&desc_handle);

		return arma::conv_to< MAT >::from(x);
	}

	/**
	 * @brief in-place 3D fast Fourier transform
	 * @param[in,out] f input-output
	 */
	void fft(arma::cx_cube &f)
	{
		const long N1 = static_cast<long>(f.n_rows);
		const long N2 = static_cast<long>(f.n_cols);
		const long N3 = static_cast<long>(f.n_slices);
		const long l[] = {N3, N2, N1};

		DFTI_DESCRIPTOR_HANDLE desc_handle;
		long status;
		status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 3, l);
		status = DftiCommitDescriptor(desc_handle);
		status = DftiComputeForward(desc_handle, f.memptr());
		status = DftiFreeDescriptor(&desc_handle);
	}

	/**
	 * @brief 3D fast Fourier transform
	 * @param[in] f input
	 * @return F Fourier coefficients
	 */
	template < class CUBE, class T >
	CUBE fft(const arma::Cube< T > &f)
	{
		const long N1 = static_cast<long>(f.n_rows);
		const long N2 = static_cast<long>(f.n_cols);
		const long N3 = static_cast<long>(f.n_slices);
		const long l[] = {N3, N2, N1};

		arma::cx_cube x = arma::conv_to< arma::cx_cube >::from(f);

		DFTI_DESCRIPTOR_HANDLE desc_handle;
		long status;
		status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 3, l);
		status = DftiCommitDescriptor(desc_handle);
		status = DftiComputeForward(desc_handle, x.memptr());
		status = DftiFreeDescriptor(&desc_handle);

		return arma::conv_to< CUBE >::from(x);
	}

	/**
	 * @brief in-place 1D inverse fast Fourier transform
	 * @param[in,out] f input-output
	 */
	void ifft(arma::cx_vec &f)
	{
		const long N = static_cast<long>(f.n_elem);

		DFTI_DESCRIPTOR_HANDLE desc_handle;
		long status;
		status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, N);
		status = DftiSetValue(desc_handle, DFTI_BACKWARD_SCALE, 1.0 / N);
		status = DftiCommitDescriptor(desc_handle);
		status = DftiComputeBackward(desc_handle, f.memptr());
		status = DftiFreeDescriptor(&desc_handle);
	}

	/**
	 * @brief 1D inverse fast Fourier transform
	 * @param[in] F Fourier coefficients
	 * @return output
	 */
	template < class COL, class T >
	COL ifft(const arma::Col< T > &F)
	{
		const long N = static_cast<long>(F.n_elem);

		arma::cx_vec x = arma::conv_to< arma::cx_vec >::from(F);

		DFTI_DESCRIPTOR_HANDLE desc_handle;
		long status;
		status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, N);
		status = DftiSetValue(desc_handle, DFTI_BACKWARD_SCALE, 1.0 / N);
		status = DftiCommitDescriptor(desc_handle);
		status = DftiComputeBackward(desc_handle, x.memptr());
		status = DftiFreeDescriptor(&desc_handle);

		return arma::conv_to< COL >::from(x);
	}

	/**
	 * @brief in-place 2D inverse fast Fourier transform
	 * @param[in,out] f input-output
	 */
	void ifft(arma::cx_mat &f)
	{
		const long N1 = static_cast<long>(f.n_rows);
		const long N2 = static_cast<long>(f.n_cols);
		const long l[] = {N2, N1};

		DFTI_DESCRIPTOR_HANDLE desc_handle;
		long status;
		status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 2, l);
		status = DftiSetValue(desc_handle, DFTI_BACKWARD_SCALE, 1.0 / (N1 * N2));
		status = DftiCommitDescriptor(desc_handle);
		status = DftiComputeBackward(desc_handle, f.memptr());
		status = DftiFreeDescriptor(&desc_handle);
	}

	/**
	 * @brief 2D inverse fast Fourier transform
	 * @param[in] F Fourier coefficients
	 * @return output
	 */
	template< class MAT, class T >
	MAT ifft(const arma::Mat< T > &F)
	{
		const long N1 = static_cast<long>(F.n_rows);
		const long N2 = static_cast<long>(F.n_cols);
		const long l[] = {N2, N1};

		arma::cx_mat x = arma::conv_to< arma::cx_mat >::from(F);

		DFTI_DESCRIPTOR_HANDLE desc_handle;
		long status;
		status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 2, l);
		status = DftiSetValue(desc_handle, DFTI_BACKWARD_SCALE, 1.0 / (N1 * N2));
		status = DftiCommitDescriptor(desc_handle);
		status = DftiComputeBackward(desc_handle, x.memptr());
		status = DftiFreeDescriptor(&desc_handle);

		return arma::conv_to< MAT >::from(x);
	}

	/**
	 * @brief in-place 3D inverse fast Fourier transform
	 * @param[in,out] f input-output
	 */
	void ifft(arma::cx_cube &f)
	{
		const long N1 = static_cast<long>(f.n_rows);
		const long N2 = static_cast<long>(f.n_cols);
		const long N3 = static_cast<long>(f.n_slices);
		const long l[] = {N3, N2, N1};

		DFTI_DESCRIPTOR_HANDLE desc_handle;
		long status;
		status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 3, l);
		status = DftiSetValue(desc_handle, DFTI_BACKWARD_SCALE, 1.0 / (N1 * N2 * N3));
		status = DftiCommitDescriptor(desc_handle);
		status = DftiComputeBackward(desc_handle, f.memptr());
		status = DftiFreeDescriptor(&desc_handle);
	}

	/**
	 * @brief 3D inverse fast Fourier transform
	 * @param[in] F Fourier coefficients
	 * @return output
	 */
	template< class CUBE, class T >
	CUBE ifft(const arma::Cube< T > &F)
	{
		const long N1 = static_cast<long>(F.n_rows);
		const long N2 = static_cast<long>(F.n_cols);
		const long N3 = static_cast<long>(F.n_slices);
		const long l[] = {N3, N2, N1};

		arma::cx_cube x = arma::conv_to< arma::cx_cube >::from(F);

		DFTI_DESCRIPTOR_HANDLE desc_handle;
		long status;
		status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 3, l);
		status = DftiSetValue(desc_handle, DFTI_BACKWARD_SCALE, 1.0 / (N1 * N2 * N3));
		status = DftiCommitDescriptor(desc_handle);
		status = DftiComputeBackward(desc_handle, x.memptr());
		status = DftiFreeDescriptor(&desc_handle);

		return arma::conv_to< CUBE >::from(x);
	}
}
