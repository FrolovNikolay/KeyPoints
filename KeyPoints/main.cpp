// Автор: Николай Фролов.
// Описание: main задания 5 по анализу изображений.

#include <ScoreCalculator.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp> // BRISK
#include <opencv2/imgproc/imgproc.hpp> // Harris
#include <opencv2/xfeatures2d/nonfree.hpp> // SIFT

using namespace std;
using namespace cv;

static void readImages( vector<Mat>& images, const string& imagesFilePath )
{
	images.clear();
	ifstream imageFile( imagesFilePath );
	vector<string> imagePaths;
	string lastPath;
	while( imageFile >> lastPath ) {
		imagePaths.push_back( lastPath );
	}
	for( const auto& imagePath : imagePaths ) {
		images.push_back( imread( imagePath, CV_LOAD_IMAGE_GRAYSCALE ) );
	}
}

static void readMotionVectors( vector<Point>& motionVectors, const string& vectorsFilePath )
{
	motionVectors.clear();
	ifstream vectorsFile( vectorsFilePath );
	double currentX;
	while( vectorsFile >> currentX ) {
		double currentY;
		vectorsFile >> currentY;
		// Вектор для точности считал нецелым. Сейчас для удобства вернем целые.
		motionVectors.push_back( Point( std::round( currentX ), std::round( currentY ) ) );
	}
}

// По заданию стремиться особо не к чему, поэтому не настраивалось - нагло спионерил из примера OpenCV.
static void harrisDetector( vector<Point>& corners, const Mat& inputImage )
{
	static const int thresh = 200;
	corners.clear();

	Mat outputImage = Mat::zeros( inputImage.size(), CV_32FC1 );

	// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;
	// Detecting corners
	cornerHarris( inputImage, outputImage, blockSize, apertureSize, k, BORDER_DEFAULT );

	// Normalizing
	Mat normalizedOutputImage;
	normalize( outputImage, normalizedOutputImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );

	// Drawing a circle around corners
	for( int j = 0; j < normalizedOutputImage.rows; j++ ) {
		for( int i = 0; i < normalizedOutputImage.cols; i++ ) {
			if( ( int )normalizedOutputImage.at<float>( j, i ) > thresh ) {
				corners.push_back( Point( i, j ) );
			}
		}
	}
}

static void siftDetector( vector<Point>& keyPoints, const Mat& inputImage )
{
	keyPoints.clear();
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
	vector<KeyPoint> keyPointsSIFT;
	f2d->detect( inputImage, keyPointsSIFT );
	for( const auto& keyPointSIFT : keyPointsSIFT ) {
		keyPoints.push_back( keyPointSIFT.pt );
	}
}

static void briskDetector( vector<Point>& keyPoints, const Mat& inputImage )
{
	keyPoints.clear();
	Ptr<Feature2D> f2d = BRISK::create();
	vector<KeyPoint> keyPointsBRISK;
	f2d->detect( inputImage, keyPointsBRISK );
	for( const auto& keyPointBRISK : keyPointsBRISK ) {
		keyPoints.push_back( keyPointBRISK.pt );
	}
}

int main( int argc, char* argv[] )
{
	if( argc != 4 ) {
		cout << "Usage: KeyPoints <ImagePaths.txt> <MotionVectors.txt> <Scores.txt>" << endl;
		system( "pause" );
		return 0;
	}

	vector<Mat> images;
	readImages( images, argv[1] );
	vector<Point> motionVectors;
	readMotionVectors( motionVectors, argv[2] );
	ofstream scores( argv[3] );

	// Harris
	vector< vector<Point> > harrisKeyPoints;
	for( auto& image : images ) {
		vector<Point> keyPoints;
		harrisDetector( keyPoints, image );
		harrisKeyPoints.push_back( keyPoints );
	}
	scores << "Harris:" << endl;
	for( size_t imageIdx = 1; imageIdx < harrisKeyPoints.size(); imageIdx++ ) {
		scores << setprecision( 3 ) << CScoreCalculator::CalculateScore( harrisKeyPoints[0], harrisKeyPoints[imageIdx], motionVectors[imageIdx - 1] ) << " ";
	}
	scores << endl << endl;
	// SIFT
	vector< vector<Point> > siftKeyPoints;
	for( auto& image : images ) {
		vector<Point> keyPoints;
		siftDetector( keyPoints, image );
		siftKeyPoints.push_back( keyPoints );
	}
	scores << "SIFT:" << endl;
	for( size_t imageIdx = 1; imageIdx < siftKeyPoints.size(); imageIdx++ ) {
		scores << setprecision( 3 ) << CScoreCalculator::CalculateScore( siftKeyPoints[0], siftKeyPoints[imageIdx], motionVectors[imageIdx - 1] ) << " ";
	}
	scores << endl << endl;
	// BRISK
	vector< vector<Point> > briskKeyPoints;
	for( auto& image : images ) {
		vector<Point> keyPoints;
		briskDetector( keyPoints, image );
		briskKeyPoints.push_back( keyPoints );
	}
	scores << "BRISK:" << endl;
	for( size_t imageIdx = 1; imageIdx < briskKeyPoints.size(); imageIdx++ ) {
		scores << setprecision( 3 ) << CScoreCalculator::CalculateScore( briskKeyPoints[0], briskKeyPoints[imageIdx], motionVectors[imageIdx - 1] ) << " ";
	}
	scores << endl << endl;

	return 0;
}