#pragma once

#include <glm/matrix.hpp>

namespace tg
{

// takes radians
glm::mat4 calcOrbitCameraMtx(glm::vec3 center, float heading, float pitch, float distance);

}
