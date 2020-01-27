#include "rt.h"

Camera make_camera(const Vec3 &location, const Vec3 &target, const Vec3 &view_up, float fov_vertical, float aspect)
{
  float half_theta = fov_vertical * M_PI / 360.0;
  float half_height = tan(half_theta);
  float half_width = aspect * half_height;

  Vec3 w = unit(location - target);
  Vec3 u = unit(cross(view_up, w));
  Vec3 v = cross(w, u);

  Vec3 lower_left_corner = location - (u * half_width) - (v * half_height) - w;
  Vec3 horizontal = u * (half_width * 2);
  Vec3 vertical = v * (half_height * 2);

  return {location, lower_left_corner, horizontal, vertical};
}

Ray get_ray(const Camera &c, float u, float v)
{
  return {c.location, c.lower_left_corner + (c.horizontal * u) + (c.vertical * v) - c.location};
}
