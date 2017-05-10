// Автор: Николай Фролов.
// Описание: Класс для расчета устойчивости детекторов ключевых точек.

#include <ScoreCalculator.h>

using namespace std;
using namespace cv;

double CScoreCalculator::CalculateScore( const std::vector<cv::Point>& keyPoints1, const std::vector<cv::Point>& keyPoints2,
	const cv::Point& motionVector )
{
	double matchedPoints = 0.0;
	for( const auto& keyPoint : keyPoints1 ) {
		if( matches( keyPoint, keyPoints2, motionVector ) ) {
			matchedPoints++;
		}
	}

	// Не уверен, на что нужно делить - на максимум/минимум/сумму. Если что, то не долго поменять.
	return matchedPoints / static_cast< double >( max( keyPoints1.size(), keyPoints2.size() ) );
}

// По заданию особой производительности не требуется, будем брать перебором.
bool CScoreCalculator::matches( const cv::Point& keyPointToMatch, const std::vector<cv::Point>& keyPoints, const cv::Point& motionVector )
{
	// Не уверен, нужно ли точно матчить? По идее у нас есть ошибка в motionVector точно матчить не хочется,
	// Но, если что, то выставляем 0, но тогда и матчей мы почти наверно увидим не много :(
	static const int maxSQEuclidDifference = 4;
	for( const auto& keyPoint : keyPoints ) {
		Point difference = keyPointToMatch + motionVector - keyPoint;
		if( difference.x * difference.x + difference.y * difference.y <= maxSQEuclidDifference ) {
			return true;
		}
	}
	return false;
}