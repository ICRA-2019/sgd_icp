#include <cstddef>

#include "fixed_sgd.h"


FixedSgd::FixedSgd(
        std::vector<double>             initial_values,
        double                          step_size
)   :   AbstractSgdOptimizer(initial_values)
      , m_step_size(step_size)
{}

void FixedSgd::do_perform_update(std::vector<double> const& gradients)
{
    for(size_t i=0; i<gradients.size(); ++i)
    {
        m_parameters[i] -= m_step_size * gradients[i];
    }
}
