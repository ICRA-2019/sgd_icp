#ifndef __POINT_CLOUD_BATCH_H__
#define __POINT_CLOUD_BATCH_H__


#include <random>

#include "types.h"


/**
 * \brief Provides random batches of points from a point cloud.
 */
class PointCloudBatch
{
    public:
        /**
         * \brief Creates a new point cloud batcher instance.
         *
         * \param point_cloud the cloud to sample batches from
         * \param batch_size number of points per batch
         */
        PointCloudBatch(Cloud_t point_cloud, int batch_size);

        /**
         * \brief Returns a new batch of points from the cloud.
         *
         * \return new batch of points
         */
        Cloud_t::Ptr next_batch();


    private:
        /**
         * \brief Randomizes the data to provide random batches.
         */
        void shuffle_data();


    private:
        //! Number of points per batch
        int                             m_batch_size;
        //! Current offset from the beginning of the cloud
        int                             m_current_offset;
        //! Point cloud which is the source of the batch data
        Cloud_t                         m_cloud;
        //! Random number generator
        std::mt19937                    m_generator;
};


#endif /* __POINT_CLOUD_BATCH_H__ */
