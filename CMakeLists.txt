cmake_minimum_required(VERSION 3.8)
project(test_ggx)

set (CMAKE_CXX_STANDARD 17)

# this function preppends a path to all files in a list
FUNCTION(PREPEND var prefix)
SET(listVar "")
   FOREACH(f ${ARGN})
   LIST(APPEND listVar "${prefix}/${f}")
   ENDFOREACH(f)
   SET(${var} "${listVar}" PARENT_SCOPE)
ENDFUNCTION(PREPEND)

add_definitions(-DGLM_FORCE_RADIANS)

find_package(OpenGL REQUIRED)
if(OPENGL_FOUND)
    include_directories(${OPENGL_INCLUDE_DIRS})
    link_libraries(${OPENGL_LIBRARIES})
endif()

add_subdirectory(libs/glad)
add_subdirectory(libs/glfw)
add_subdirectory(libs/imgui)
add_subdirectory(libs/stb)
add_subdirectory(libs/glm)
add_subdirectory(libs/tl)
add_subdirectory(libs/tg)

add_executable(test_ggx
    main.cpp
    utils.hpp
)
target_link_libraries(test_ggx glad)
target_link_libraries(test_ggx glfw)
target_link_libraries(test_ggx imgui)
target_link_libraries(test_ggx stb)
target_link_libraries(test_ggx glm)
target_link_libraries(test_ggx tl)
target_link_libraries(test_ggx tg)

add_executable(prefilter
    prefilter.cpp
    utils.hpp
)
target_link_libraries(prefilter glad)
target_link_libraries(prefilter glfw)
target_link_libraries(prefilter imgui)
target_link_libraries(prefilter stb)
target_link_libraries(prefilter glm)
target_link_libraries(prefilter tl)
target_link_libraries(prefilter tg)

