// Автор: Николай Фролов.
// Описание: Класс для расчета устойчивости детекторов ключевых точек.

#include <vector>
#include <opencv2/core/core.hpp>

#pragma once

class CScoreCalculator {
public:
	// MotionVector должен быть от изображения с точками keyPoints1 к изображению с точками keyPoints2.
	static double CalculateScore( const std::vector<cv::Point>& keyPoints1, const std::vector<cv::Point>& keyPoints2,
		const cv::Point& motionVector );
private:
	// Проверяет есть ли среди keyPoints соответствующая keyPoint точка.
	static bool matches( const cv::Point& keyPointToMatch, const std::vector<cv::Point>& keyPoints, const cv::Point& motionVector );
};