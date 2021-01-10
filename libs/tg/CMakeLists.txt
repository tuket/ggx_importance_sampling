project(tg)

add_library(tg
	img.hpp img.cpp
	shader_utils.hpp shader_utils.cpp
	mesh_utils.hpp mesh_utils.cpp
	cameras.hpp cameras.cpp
)

target_include_directories(tg PUBLIC  ${PROJECT_SOURCE_DIR}/..)

target_link_libraries(tg
	glm
	stb
	tl
	glad
)