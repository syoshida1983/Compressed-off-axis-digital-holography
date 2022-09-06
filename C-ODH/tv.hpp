/**
 * @file tv.hpp
 * @brief functions for total variation regularization with (fast) gradient projection
 * @details for more information on fast gradient projection, see https://doi.org/10.1109/TIP.2009.2028250
 */

#pragma once

#include <armadillo>

/**
 * @namespace cs
 * @brief compressed sensing
 */
namespace cs
{
	using namespace std;

	/**
	 * @namespace tv
	 * @brief total variation regularization
	 */
	namespace tv
	{
		/**
		 * @brief linear operation L
		 * @param[in]  p dual variable
		 * @param[in]  q dual variable
		 * @param[out] x primal variable
		 */
		template < class T >
		void linearop(
			const	arma::Mat< T > &p,
			const	arma::Mat< T > &q,
					arma::Mat< T > &x)
		{
			const auto N1 = p.n_rows;
			const auto N2 = p.n_cols;

			x.zeros();

			#pragma omp parallel for
			for (auto j = 1; j < N2; j++)
			{
				for (auto i = 1; i < N1; i++)
				{
					x(i, j) = p(i, j) + q(i, j) - p(i - 1, j) - q(i, j - 1);
				}
			}
		}

		/**
		 * @brief adjoint operator to L
		 * @param[in]  x primal variable
		 * @param[out] p dual variable
		 * @param[out] q dual variable
		 */
		template < class T >
		void adjointop(
			const	arma::Mat< T > &x,
					arma::Mat< T > &p,
					arma::Mat< T > &q)
		{
			const auto N1 = x.n_rows;
			const auto N2 = x.n_cols;

			p.zeros();
			q.zeros();

			#pragma omp parallel for
			for (auto j = 0; j < N2 - 1; j++)
			{
				for (auto i = 0; i < N1 - 1; i++)
				{
					p(i, j) = x(i, j) - x(i + 1, j);
					q(i, j) = x(i, j) - x(i, j + 1);
				}
			}
		}

		/**
		 * @brief projection onto set of matrix-pairs (p,q)
		 * @param[in]  p dual variable
		 * @param[in]  q dual variable
		 * @param[out] r working variable for p
		 * @param[out] s working variable for q
		 */
		template < class T >
		void projection(
			const	arma::Mat< T > &p,
			const	arma::Mat< T > &q,
					arma::Mat< T > &r,
					arma::Mat< T > &s)
		{
			const auto N1 = p.n_rows;
			const auto N2 = p.n_cols;

			#pragma omp parallel for
			for (auto j = 0; j < N2; j++)
			{
				for (auto i = 0; i < N1; i++)
				{
					const double d = std::max(1.0, sqrt(norm(p(i, j)) + norm(q(i, j))));
					r(i, j) = p(i, j) / d;
					s(i, j) = q(i, j) / d;
				}
			}
		}

		/**
		 * @brief fast gradient projection
		 * @param[in] v      input variable
		 * @param[in] lambda regularization parameter
		 * @param[in] N      number of iterations
		 * @return minimized variable
		 */
		template < class MAT, class T >
		MAT gp(
			const arma::Mat< T >	&v,
			const double			lambda,
			const long long			N)
		{
			const auto N1 = v.n_rows;
			const auto N2 = v.n_cols;

			MAT x(N1, N2);
			arma::Mat< T > p(N1, N2, arma::fill::zeros);
			arma::Mat< T > q(N1, N2, arma::fill::zeros);
			arma::Mat< T > r(N1, N2, arma::fill::zeros);
			arma::Mat< T > s(N1, N2, arma::fill::zeros);

			for (auto i = 0; i < N; i++)
			{
				cs::tv::linearop(r, s, x);
				x = v - lambda * x;
				cs::tv::adjointop(x, p, q);
				p = r + p / (8 * lambda);
				q = s + q / (8 * lambda);
				cs::tv::projection(p, q, r, s);
			}

			cs::tv::linearop(r, s, x);
			return v - lambda * x;
		}

		/**
		 * @brief fast gradient projection
		 * @param[in] v      input variable
		 * @param[in] lambda regularization parameter
		 * @param[in] N      number of iterations
		 * @return minimized variable
		 */
		template < class MAT, class T >
		MAT fgp(
			const arma::Mat< T >	&v,
			const double			lambda,
			const long long			N)
		{
			const auto N1 = v.n_rows;
			const auto N2 = v.n_cols;

			MAT x(N1, N2);
			arma::Mat< T > n(N1, N2, arma::fill::zeros);
			arma::Mat< T > o(N1, N2, arma::fill::zeros);
			arma::Mat< T > p(N1, N2, arma::fill::zeros);
			arma::Mat< T > q(N1, N2, arma::fill::zeros);
			arma::Mat< T > r(N1, N2, arma::fill::zeros);
			arma::Mat< T > s(N1, N2, arma::fill::zeros);
			double a = 1;
			double b = 1;

			for (auto i = 0; i < N; i++)
			{
				cs::tv::linearop(r, s, x);
				x = v - lambda * x;
				cs::tv::adjointop(x, r, s);
				r = p + r / (8 * lambda);
				s = q + s / (8 * lambda);
				cs::tv::projection(r, s, p, q);
				b = (1 + sqrt(1 + 4 * a * a)) / 2;
				r = p + ((a - 1) / b) * (p - n);
				s = q + ((a - 1) / b) * (q - o);
				n = p;
				o = q;
				p = r;
				q = s;
				a = b;
			}

			cs::tv::linearop(r, s, x);
			return v - lambda * x;
		}

		/**
		 * @brief gradient projection for each layer of volume
		 * @param[in] V      input variable
		 * @param[in] lambda regularization parameter
		 * @param[in] N      number of iterations
		 * @return minimized variable
		 */
		template < class CUBE, class T >
		CUBE gp(
			const arma::Cube< T >	&V,
			const double			lambda,
			const long long			N)
		{
			const auto N1 = V.n_rows;
			const auto N2 = V.n_cols;
			const auto N3 = V.n_slices;

			CUBE X(N1, N2, N3);

			#pragma omp parallel for
			for (auto i = 0; i < N3; i++)
			{
				X.slice(i) = cs::tv::gp< arma::Mat< typename CUBE::elem_type > >(V.slice(i), lambda, N);
			}

			return X;
		}

		/**
		 * @brief fast gradient projection for each layer of volume
		 * @param[in] V      input variable
		 * @param[in] lambda regularization parameter
		 * @param[in] N      number of iterations
		 * @return minimized variable
		 */
		template < class CUBE, class T >
		CUBE fgp(
			const arma::Cube< T >	&V,
			const double			lambda,
			const long long			N)
		{
			const auto N1 = V.n_rows;
			const auto N2 = V.n_cols;
			const auto N3 = V.n_slices;

			CUBE X(N1, N2, N3);

			#pragma omp parallel for
			for (auto i = 0; i < N3; i++)
			{
				X.slice(i) = cs::tv::fgp< arma::Mat< typename CUBE::elem_type > >(V.slice(i), lambda, N);
			}

			return X;
		}
	}
}
