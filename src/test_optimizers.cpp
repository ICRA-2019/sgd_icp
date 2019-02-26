#include <cmath>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "adam.h"
#include "adadelta.h"
#include "fixed_sgd.h"
#include "rmsprop.h"

double f(double x)
{
    return 5*x*x + 2*x + 3;
}

double fdx(double x)
{
    return 10*x + 2;
}


TEST_CASE("adam", "")
{
    double param = 10.0;
    auto adam = Adam({param}, 0.1, 0.9, 0.999);
    for(int i=0; i<1000; ++i)
    {
        auto new_param = adam.update_parameters({fdx(param)});
        param = new_param[0];
    }
    REQUIRE(-0.2 == Approx(param).margin(0.00001));
}

TEST_CASE("adadelta", "")
{
    double param = 10.0;
    auto adadelta = AdaDelta({param}, 0.9, 1e-3);
    for(int i=0; i<1000; ++i)
    {
        auto new_param = adadelta.update_parameters({fdx(param)});
        param = new_param[0];
    }
    REQUIRE(-0.2 == Approx(param).margin(0.00001));
}

TEST_CASE("fixed", "")
{
    double param = 10.0;
    auto fixed = FixedSgd({param}, 0.1);
    for(int i=0; i<1000; ++i)
    {
        auto new_param = fixed.update_parameters({fdx(param)});
        param = new_param[0];
    }
    REQUIRE(-0.2 == Approx(param).margin(0.00001));
}

TEST_CASE("rmsprop", "")
{
    double param = 10.0;
    auto rmsprop = Rmsprop({param}, 0.01, 0.9);
    for(int i=0; i<1000; ++i)
    {
        auto new_param = rmsprop.update_parameters({fdx(param)});
        param = new_param[0];
    }
    REQUIRE(-0.2 == Approx(param).margin(0.00001));
}
