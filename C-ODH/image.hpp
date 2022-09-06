/**
 * @file image.hpp
 * @brief functions of image input/output using OpenCV for Armadillo
 */

#pragma once

#include <string>
#include <map>
#include <boost/math/special_functions.hpp>
#include <opencv2/opencv.hpp>
#include <armadillo>

/**
 * @namespace image
 * @brief input/output of images to/from Armadillo matrix using OpenCV
 */
namespace image
{
	using namespace std;

	/**
	 * @brief read grayscale (8-bit)/full color (24-bit) image
	 * @param[in] filename file name
	 * @return std::map containing Armadillo matrices
	 * @details in the case of a grayscale image, the Armadillo matrix can be retrieved by .at("k")
	 * @details in the case of a full color image, the matrix of each channel can be obtained by setting the key to "r", "g", or "b"
	 */
	template < class MAT >
	map< string, MAT > imread(const string& filename)
	{
		cv::Mat img = cv::imread(filename, cv::IMREAD_UNCHANGED);
		CV_Assert(img.type() == CV_8UC1 || img.type() == CV_8UC3);

		map< string, MAT > m;

		if (img.channels() == 1)
		{
			MAT k(img.size().height, img.size().width);

			for (auto j = 0; j < img.size().width; j++)
			{
				for (auto i = 0; i < img.size().height; i++)
				{
					k(i, j) = static_cast<typename MAT::elem_type>(img.at< uchar >(i, j));
				}
			}

			m.insert(make_pair("k", k));
		}
		else if (img.channels() == 3)
		{
			MAT b(img.size().height, img.size().width);
			MAT g(img.size().height, img.size().width);
			MAT r(img.size().height, img.size().width);

			for (auto j = 0; j < img.size().width; j++)
			{
				for (auto i = 0; i < img.size().height; i++)
				{
					b(i, j) = static_cast<typename MAT::elem_type>(img.at<cv::Vec3b>(i, j)[0]);
					g(i, j) = static_cast<typename MAT::elem_type>(img.at<cv::Vec3b>(i, j)[1]);
					r(i, j) = static_cast<typename MAT::elem_type>(img.at<cv::Vec3b>(i, j)[2]);
				}
			}

			m.insert(make_pair("r", r));
			m.insert(make_pair("g", g));
			m.insert(make_pair("b", b));
		}

		return m;
	}

	/**
	 * @brief write grayscale (8-bit) image
	 * @param[in] filename file name
	 * @param[in] k        Armadillo matrix
	 * @return success or failure
	 * @details normalized with a maximum value of 255 and a minimum value of 0
	 */
	template < class MAT >
	bool imwrite(const string& filename, MAT& k)
	{
		cv::Mat img(k.n_rows, k.n_cols, CV_8UC1);

		const auto max = k.max();
		const auto min = k.min();
		const auto range = max - min;

		for (auto j = 0; j < img.size().width; j++)
		{
			for (auto i = 0; i < img.size().height; i++)
			{
				img.at< uchar >(i, j) = static_cast<uchar>(boost::math::iround(255 * (k(i, j) - min) / range));
			}
		}

		return cv::imwrite(filename, img);
	}

	/**
	 * @brief write grayscale (8-bit) image
	 * @param[in] filename file name
	 * @param[in] k        Armadillo matrix
	 * @param[in] max      maximum value for normalization
	 * @param[in] min      minimum value for normalization
	 * @return success or failure
	 * @details normalized with argument "max" of 255 and argument "min" of 0
	 */
	template < class MAT >
	bool imwrite(const string& filename, MAT& k, typename MAT::elem_type max, typename MAT::elem_type min)
	{
		cv::Mat img(k.n_rows, k.n_cols, CV_8UC1);

		const auto range = max - min;

		for (auto j = 0; j < img.size().width; j++)
		{
			for (auto i = 0; i < img.size().height; i++)
			{
				img.at< uchar >(i, j) = static_cast<uchar>(boost::math::iround(255 * (k(i, j) - min) / range));
			}
		}

		return cv::imwrite(filename, img);
	}

	/**
	 * @brief write full color (24-bit) image
	 * @param[in] filename file name
	 * @param[in] r        Armadillo matrix containing R channel data
	 * @param[in] g        Armadillo matrix containing G channel data
	 * @param[in] b        Armadillo matrix containing B channel data
	 * @return success or failure
	 * @details normalized with a maximum value of all channels of 255 and a minimum value of 0
	 */
	template < class MAT >
	bool imwrite(const string& filename, MAT& r, MAT& g, MAT& b)
	{
		cv::Mat img(r.n_rows, r.n_cols, CV_8UC3);

		const auto max = std::max({ r.max(), g.max(), b.max() });
		const auto min = std::min({ r.min(), g.min(), b.min() });
		const auto range = max - min;

		for (auto j = 0; j < img.size().width; j++)
		{
			for (auto i = 0; i < img.size().height; i++)
			{
				img.at<cv::Vec3b>(i, j)[0] = static_cast<uchar>(boost::math::iround(255 * (b(i, j) - min) / range));
				img.at<cv::Vec3b>(i, j)[1] = static_cast<uchar>(boost::math::iround(255 * (g(i, j) - min) / range));
				img.at<cv::Vec3b>(i, j)[2] = static_cast<uchar>(boost::math::iround(255 * (r(i, j) - min) / range));
			}
		}

		return cv::imwrite(filename, img);
	}

	/**
	 * @brief write full color (24-bit) image
	 * @param[in] filename file name
	 * @param[in] r        Armadillo matrix containing R channel data
	 * @param[in] g        Armadillo matrix containing G channel data
	 * @param[in] b        Armadillo matrix containing B channel data
	 * @param[in] max      maximum value for normalization
	 * @param[in] min      minimum value for normalization
	 * @return success or failure
	 * @details normalized with argument "max" of 255 and argument "min" of 0
	 */
	template < class MAT >
	bool imwrite(const string& filename, MAT& r, MAT& g, MAT& b, typename MAT::elem_type max, typename MAT::elem_type min)
	{
		cv::Mat img(r.n_rows, r.n_cols, CV_8UC3);

		const auto range = max - min;

		for (auto j = 0; j < img.size().width; j++)
		{
			for (auto i = 0; i < img.size().height; i++)
			{
				img.at<cv::Vec3b>(i, j)[0] = static_cast<uchar>(boost::math::iround(255 * (b(i, j) - min) / range));
				img.at<cv::Vec3b>(i, j)[1] = static_cast<uchar>(boost::math::iround(255 * (g(i, j) - min) / range));
				img.at<cv::Vec3b>(i, j)[2] = static_cast<uchar>(boost::math::iround(255 * (r(i, j) - min) / range));
			}
		}

		return cv::imwrite(filename, img);
	}
}
