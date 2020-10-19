#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file

#include <GradientDescent.h>
#include "catch.hpp"

TEST_CASE("Testing Gradient Descent Algorithm", "[GradientDescent]"){
    GradientDescent solver;

    SECTION( "f(x) = x^2 using Approximate Derivatives" ) {
        auto objFunc1 = [](std::vector<double> x) {return x[0] * x[0]; };
        solver.setObjectiveFunction(objFunc1);
        std::vector<double> startPoint = { -1.0 };
        solver.setUseAAD(false);
        solver.setStartPoint(startPoint);
        solver.setMaxIterations(50);
        solver.setStepSize(0.2);
        std::pair< std::vector<double>, double> results = solver.optimize();
        double funcLoc = results.first[0];
        double funcVal = results.second;
        std::cout << "f(x) = x ^2 " << std::endl;
        std::cout << "Without AAD : Approximate Derivatives " << std::endl;
        solver.printResults(results);
        std::cout << "---------------------------------------" << std::endl;
        REQUIRE(funcLoc == Approx(0).margin(1e-3));
        REQUIRE(funcVal == Approx(0).margin(1e-3));
    }

    SECTION( "f(x) = x^2 using Exact Derivatives" ) {
        auto objFunc1 = [](std::vector<var> x) {return x[0] * x[0]; };
        solver.setObjectiveFunctionAAD(objFunc1);
        std::vector<double> startPoint = { -1.0 };
        solver.setUseAAD(true);
        solver.setStartPoint(startPoint);
        solver.setMaxIterations(50);
        solver.setStepSize(0.2);
        std::pair< std::vector<double>, double> results = solver.optimize();
        double funcLoc = results.first[0];
        double funcVal = results.second;
        std::cout << "f(x) = x^2" << std::endl;
        std::cout << "With AAD : Exact Derivatives " << std::endl;
        solver.printResults(results);
        std::cout << "---------------------------------------" << std::endl;
        REQUIRE(funcLoc == Approx(0).margin(1e-8));
        REQUIRE(funcVal == Approx(0).margin(1e-8));
    }

    SECTION( "f(x) = f(x) = x^4 + 2x^3 - 6x^2 + 4x + 2 using Approximate Derivatives" ) {
        auto objFunc2 = [](std::vector<double> x) {return (x[0] * x[0] * x[0] * x[0]) + (2 * (x[0] * x[0] * x[0])) - (6 * (x[0] * x[0])) + (4 * x[0]) + 2; };
        solver.setObjectiveFunction(objFunc2);
        std::vector<double> startPoint = { -4.0 };
        solver.setUseAAD(false);
        solver.setStartPoint(startPoint);
        solver.setStepSize(0.04);
        solver.setMaxIterations(50);
        std::pair< std::vector<double>, double> results = solver.optimize();
        double funcLoc = results.first[0];
        double funcVal = results.second;
        std::cout << "f(x) = x^4 + 2x^3 - 6x^2 + 4x + 2" << std::endl;
        std::cout << "Without AAD : Approximate Derivatives" << std::endl;
        solver.printResults(results);
        std::cout << "---------------------------------------" << std::endl;
        REQUIRE(funcLoc == Approx(-2.73257).margin(1e-3));
        REQUIRE(funcVal == Approx(-38.7846).margin(1e-3));
    }

    SECTION( "f(x) = f(x) = x^4 + 2x^3 - 6x^2 + 4x + 2 using Exact Derivatives" ) {
        auto objFunc2AAD = [](std::vector<autodiff::var> x) {return (x[0] * x[0] * x[0] * x[0]) + (2 * (x[0] * x[0] * x[0])) - (6 * (x[0] * x[0])) + (4 * x[0]) + 2; };
        solver.setObjectiveFunctionAAD(objFunc2AAD);
        std::vector<double> startPoint = { -4.0 };
        solver.setUseAAD(true);
        solver.setStartPoint(startPoint);
        solver.setStepSize(0.04);
        solver.setMaxIterations(50);
        std::pair< std::vector<double>, double> results = solver.optimize();
        double funcLoc = results.first[0];
        double funcVal = results.second;
        std::cout << "f(x) = x^4 + 2x^3 - 6x^2 + 4x + 2" << std::endl;
        std::cout << "With AAD : Exact Derivatives " << std::endl;
        solver.printResults(results);
        std::cout << "---------------------------------------" << std::endl;
        REQUIRE(funcLoc == Approx(-2.73207).margin(1e-8));
        REQUIRE(funcVal == Approx(-38.7846).margin(1e-8));
    }

    SECTION( "f(x,y) = x^2 + xy + y^2 using Approximate Derivatives" ) {
        auto objFunc3 = [](std::vector<double> x) {return (x[0] * x[0]) + (x[0] * x[1]) + (x[1] * x[1]); };
        solver.setObjectiveFunction(objFunc3);
        std::vector<double> startPoint = { 5.0,5.0 };
        solver.setUseAAD(false);
        solver.setStartPoint(startPoint);
        solver.setMaxIterations(50);
        solver.setStepSize(0.1);
        std::pair< std::vector<double>, double> results = solver.optimize();
        std::cout << "f(x,y) = x^2 + xy + y^2" << std::endl;
        std::cout << "Without AAD : Approximate Derivatives" << std::endl;
        solver.printResults(results);
        std::cout << "---------------------------------------" << std::endl;
        for (double d : results.first) {
            REQUIRE(d == Approx(0).margin(1e-3));
        }
        REQUIRE(results.second == Approx(0).margin(1e-3));
    }

    SECTION( "f(x,y) = x^2 + xy + y^2 using Exact Derivatives" ) {
        auto objFunc3AAD = [](std::vector<var> x) {return (x[0] * x[0]) + (x[0] * x[1]) + (x[1] * x[1]); };
        solver.setObjectiveFunctionAAD(objFunc3AAD);
        std::vector<double> startPoint = { 5.0,5.0 };
        solver.setUseAAD(true);
        solver.setStartPoint(startPoint);
        solver.setMaxIterations(50);
        solver.setStepSize(0.1);
        std::pair< std::vector<double>, double> results = solver.optimize();
        std::cout << "f(x,y) = x^2 + xy + y^2" << std::endl;
        std::cout << "With AAD : Exact Derivatives" << std::endl;
        solver.printResults(results);
        std::cout << "---------------------------------------" << std::endl;
        for (double d : results.first) {
            REQUIRE(d == Approx(0).margin(1e-5));
        }
        REQUIRE(results.second == Approx(0).margin(1e-8));
    }



}

