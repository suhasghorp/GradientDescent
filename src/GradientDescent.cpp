#include "GradientDescent.h"
#include <iostream>
#include <fstream>
#include <utility>
#include <cmath>
#include "autodiff/reverse.hpp"

using namespace autodiff;


GradientDescent::GradientDescent() {
    m_nDims = 0;
    m_stepSize = 0.0;
    m_maxIter = 1;
    m_h = 0.001;
    m_gradientThreshold = 1.0e-09;

}

GradientDescent::~GradientDescent()= default;

void GradientDescent::setObjectiveFunction(std::function<double(std::vector<double>)> objFun) {
    m_objFun = std::move(objFun);
}

void GradientDescent::setObjectiveFunctionAAD(std::function<var(std::vector<var>)> objFun) {
    m_objFunAAD = std::move(objFun);
}

void GradientDescent::setUseAAD(bool useAAD) {
    m_useAAD = useAAD;
}

void GradientDescent::setStartPoint(const std::vector<double>& startPoint) {
    m_startPoint = startPoint;
    m_nDims = m_startPoint.size();
}
void GradientDescent::setStepSize(double stepSize) {
    m_stepSize = stepSize;
}
void GradientDescent::setMaxIterations(int maxIterations) {
    m_maxIter = maxIterations;
}
void GradientDescent::setGradientThreshold(double gradThreshold) {
    m_gradientThreshold = gradThreshold;
}
std::vector<var> convertVector(const std::vector<double>& point) {
    std::vector<var> varPoint;
    varPoint.reserve(point.size());
    for (double d : point) {
        varPoint.emplace_back(var(d));
    }
    return varPoint;
}
std::pair< std::vector<double>, double> GradientDescent::optimize() {
    m_currentPoint = m_startPoint;

    int iterCount = 0;
    double gradientMagnitude = 1.0;
    while ((iterCount < m_maxIter) && (gradientMagnitude > m_gradientThreshold)) {
        std::vector<double> gradientVector;
        if (m_useAAD) {
            gradientVector = computeGradientVectorAAD();
        } else {
            gradientVector = computeGradientVector();
        }
        gradientMagnitude = computeGradientMagnitude(gradientVector);


        std::vector<double> newPoint = m_currentPoint;
        for (int i = 0; i < m_nDims; ++i) {
            newPoint[i] += -(gradientVector[i] * m_stepSize);
        }

        m_currentPoint = newPoint;
        iterCount++;
    }

    std::pair< std::vector<double>, double> results;
    results.first = m_currentPoint;
    if (m_useAAD) {

        results.second = double(m_objFunAAD(convertVector(m_currentPoint)));
    } else {
        results.second = m_objFun(m_currentPoint);
    }

    return results;
}



void GradientDescent::printResults(const std::pair< std::vector<double>, double>& results) {
    std::cout << "Function Location: ";
    for (double d : results.first) {
        std::cout << d << "   ";
    }
    std::cout << std::endl;
    std::cout << "Function Value: " << results.second << std::endl;
}



double GradientDescent::computeGradient(int dim) {

    std::vector<double> newPoint = m_currentPoint;
    newPoint[dim] += m_h;

    double funcVal1 = m_objFun(m_currentPoint);
    double funcVal2 = m_objFun(newPoint);

    return (funcVal2 - funcVal1) / m_h;

}

std::vector<double> GradientDescent::computeGradientVectorAAD() {
    std::vector<double> gradientVector;
    if (m_nDims > 3)
        throw std::runtime_error(std::string("We support only 3 dimensions: f(x), f(x,y) and f(x,y,z) and that's it..."));
    if (m_nDims == 1) {
        var x = m_currentPoint[0];
        std::vector<var> vecx{ x };
        var u = m_objFunAAD(vecx);
        auto [ux] = derivatives(u, wrt(x));
        gradientVector.push_back(double(ux));
    } else if (m_nDims == 2) {
        var x = m_currentPoint[0];
        var y = m_currentPoint[1];
        std::vector<var> vecx{ x , y};
        var u = m_objFunAAD(vecx);
        auto [ux, uy] = derivatives(u, wrt(x,y));
        gradientVector.push_back(double(ux));
        gradientVector.push_back(double(uy));
    } else if (m_nDims == 3) {
        var x = m_currentPoint[0];
        var y = m_currentPoint[1];
        var z = m_currentPoint[2];
        std::vector<var> vecx{ x , y, z};
        var u = m_objFunAAD(vecx);
        auto [ux, uy, uz] = derivatives(u, wrt(x,y,z));
        gradientVector.push_back(double(ux));
        gradientVector.push_back(double(uy));
        gradientVector.push_back(double(uz));
    }
    return gradientVector;
}

std::vector<double> GradientDescent::computeGradientVector() {
    std::vector<double> gradientVector = m_currentPoint;
    for (int i = 0; i < m_nDims; ++i) {
        gradientVector[i] = computeGradient(i);
    }
    return gradientVector;
}

double GradientDescent::computeGradientMagnitude(std::vector<double> gradientVector) const {
    double vectorMagnitude = 0.0;
    for (int i = 0; i < m_nDims; ++i) {
        vectorMagnitude += gradientVector[i] * gradientVector[i];
    }
    return sqrt(vectorMagnitude);

}