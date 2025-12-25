set(_LIB_NAME csaps)

set(_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/csaps.h
    ${CMAKE_CURRENT_SOURCE_DIR}/csaps.cpp
)

add_library(${_LIB_NAME} STATIC ${_SOURCES})
target_include_directories(${_LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/csaps>
  ${EIGEN3_INCLUDE_DIR}
)
target_compile_features(${_LIB_NAME} PUBLIC cxx_std_11)

# 创建别名
add_library(csaps::csaps ALIAS csaps)

install(TARGETS csaps
  EXPORT csapsTargets
  DESTINATION lib
)

install(FILES
    csaps.h
    DESTINATION include/csaps
)
