#set(LIB_EMBREE_DIR "${LIB_DIR}/embree/")

#set(EMBREE_TUTORIALS OFF CACHE INTERNAL "Enable to build Embree tutorials")

#add_subdirectory(${LIB_EMBREE_DIR})
find_package(Embree 3.12.1 REQUIRED)
include_directories(${EMBREE_INCLUDE_DIRS})