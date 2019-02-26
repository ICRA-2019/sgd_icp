#include <cmath>
#include <stddef.h>

#include "adadelta.h"


AdaDelta::AdaDelta(
        std::vector<double>             initial_values,
        double                          decay_rate,
        double                          preconditioner
)
    :   AbstractSgdOptimizer(initial_values)
      , m_decay_rate(decay_rate)
      , m_preconditioner(preconditioner)
      , m_acc_gradient(initial_values.size(), 0.0)
      , m_acc_updates(initial_values.size(), 0.0)
{}

void AdaDelta::do_perform_update(std::vector<double> const& gradients)
{
    auto decay_inv = 1.0 - m_decay_rate;

    for(size_t i=0; i<gradients.size(); ++i)
    {
        // Accumulate gradient via exponential decay
        m_acc_gradient[i] = m_decay_rate * m_acc_gradient[i] +
            decay_inv * gradients[i] * gradients[i];

        // Compute RMS values
        double rms_gradient = std::sqrt(m_acc_gradient[i] + m_preconditioner);
        double rms_update = std::sqrt(m_acc_updates[i] + m_preconditioner);

        // Compute update value
        double update = -(rms_update / rms_gradient) * gradients[i];

        // Accumulate change via exponential decay
        m_acc_updates[i] = m_decay_rate * m_acc_updates[i] +
            decay_inv * update * update;

        // Update parameter value
        m_parameters[i] += update;
    }
}
