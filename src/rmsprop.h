
#ifndef __RMSPROP_H__
#define __RMSPROP_H__


#include "abstract_sgd_optimizer.h"


class Rmsprop : public AbstractSgdOptimizer
{
public:
    Rmsprop(
         std::vector<double>         initial_values,
         double                      step_size,
         double                      decay_rate
        
         );
    
protected:
    void do_perform_update(std::vector<double> const& gradients) override;
    
private:
    double                          m_step_size;
    double                          m_decay_rate;
    
   
    std::vector<double>             m_second_moment;
};


#endif /* __RMSPROP_H__ */
