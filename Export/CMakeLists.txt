cmake_minimum_required(VERSION 2.8)

IF(DEFINED PROJECT_NAME)
   set(internal TRUE)
ENDIF(DEFINED PROJECT_NAME)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

if(internal)
	# inside rtabmap project (see below for external build)
	SET(RTABMap_INCLUDE_DIRS 
	    ${PROJECT_SOURCE_DIR}/utilite/include
		${PROJECT_SOURCE_DIR}/corelib/include
	)
	SET(RTABMap_LIBRARIES 
	    rtabmap_core
		rtabmap_utilite
	)  
else()
	# external build
	PROJECT( MyProject )
	
	FIND_PACKAGE(RTABMap REQUIRED)
	FIND_PACKAGE(OpenCV REQUIRED)
	FIND_PACKAGE(PCL 1.7 REQUIRED)
	FIND_PACKAGE(Open3D REQUIRED)
        FIND_PACKAGE(G2O REQUIRED)
	
endif()

if(POLICY CMP0020)
	cmake_policy(SET CMP0020 OLD)
endif()

SET(INCLUDE_DIRS
	${RTABMap_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${Open3D_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIR}
    include
)

SET(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

SET(G2O_LIBS g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension )

SET(LIBRARIES
	${RTABMap_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PCL_LIBRARIES}
        ${Open3D_LIBRARIES}
        ${G2O_LIBS}
)

include_directories(${INCLUDE_DIRS})

ADD_EXECUTABLE(export main.cpp)
ADD_EXECUTABLE(exportPCL src/descriptor.cpp src/elas.cpp src/filter.cpp src/matrix.cpp src/triangle.cpp exportRT.cpp)
ADD_EXECUTABLE(PointCloudMatching PointCloudMatching.cpp)
ADD_EXECUTABLE(SiftMatching SiftMatching.cpp)
ADD_EXECUTABLE(imageRecorder imageRecorder.cpp)
ADD_EXECUTABLE(directSemidense directSemidense.cpp)
ADD_EXECUTABLE(pointMosaic pointMosaic.cpp)
  
TARGET_LINK_LIBRARIES(export ${LIBRARIES})
TARGET_LINK_LIBRARIES(exportPCL ${LIBRARIES})
TARGET_LINK_LIBRARIES(PointCloudMatching ${LIBRARIES})
TARGET_LINK_LIBRARIES(SiftMatching ${LIBRARIES})
TARGET_LINK_LIBRARIES(imageRecorder ${LIBRARIES})
TARGET_LINK_LIBRARIES(directSemidense ${LIBRARIES})
TARGET_LINK_LIBRARIES(pointMosaic ${LIBRARIES})

if(internal)
	SET_TARGET_PROPERTIES( export 
	  PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-export)
endif(internal)

if(internal)
	SET_TARGET_PROPERTIES( exportPCL 
	  PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-exportPCL)
endif(internal)

if(internal)
        SET_TARGET_PROPERTIES( PointCloudMatching
          PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-PointCloudMatching)
endif(internal)

if(internal)
        SET_TARGET_PROPERTIES( SiftMatching
          PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-SiftMatching)
endif(internal)

if(internal)
        SET_TARGET_PROPERTIES( imageRecorder
          PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-imageRecorder)
endif(internal)

if(internal)
        SET_TARGET_PROPERTIES( directSemidense
          PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-directSemidense)
endif(internal)


