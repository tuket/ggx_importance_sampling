#include "cameras.hpp"

#define GLM_FOCE_RADIANS
#include <glm/gtx/euler_angles.hpp>

namespace tg
{

glm::mat4 calcOrbitCameraMtx(glm::vec3 center, float heading, float pitch, float distance)
{
    // (C * Ry * Rx * D)^-1 = D^-1 * Rx^-1 * Ry^-1 * C^-1
    auto mtx = glm::translate(glm::mat4(1), {0, 0, -distance});
    mtx *= glm::eulerAngleXY(-pitch, -heading);
    mtx = glm::translate(mtx, -center);
    return mtx;
}

}
