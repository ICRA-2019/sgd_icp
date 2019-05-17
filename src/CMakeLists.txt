# SGD-ICP Library
add_library(
    sgdicp
    SHARED
    abstract_sgd_optimizer.cpp
    adadelta.cpp
    adam.cpp
    fixed_sgd.cpp
    rmsprop.cpp
    momentum.cpp
    point_cloud_batch.cpp
    sgdicp.cpp
    utils.cpp
)
target_link_libraries(
    sgdicp
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
)

# Align two PCD clouds using libsgdicp
add_executable( align_pcd align_pcd.cpp )
target_link_libraries( align_pcd sgdicp )

# PCL based PCD alignment
add_executable( pcl_align_pcd pcl_align_pcd.cpp )
target_link_libraries(
    pcl_align_pcd
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
)

# Tool to create transformed cloud from arbitrrary inputs
add_executable( transform_cloud transform_cloud )
target_link_libraries(
    transform_cloud
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
)

# Optimizer test
add_executable( test_optimizers test_optimizers.cpp )
target_link_libraries( test_optimizers sgdicp )
