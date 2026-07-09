#include "custom_pde.h"

#include <prismspf/core/initial_conditions.h>
#include <prismspf/core/nonuniform_dirichlet.h>

#include <prismspf/user_inputs/user_input_parameters.h>

#include <prismspf/config.h>

#include <algorithm>
#include <cmath>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::set_initial_condition(
  [[maybe_unused]] const unsigned int       &index,
  [[maybe_unused]] const unsigned int       &component,
  [[maybe_unused]] const dealii::Point<dim> &point,
  [[maybe_unused]] number                   &scalar_value,
  [[maybe_unused]] number                   &vector_component_value) const
{
  const std::array<double, 3> center = {
    {0.5, 0.5, 0.5}
  };

  // Outer hollow-prism size.
  // outer_radius controls the size of the hexagonal prism in the x-y plane.
  // outer_half_height controls the half-thickness in the z direction.
  const double outer_radius      = 18.0;
  const double outer_half_height = 8.0;

  // Inner hollow-core size.
  // inner_radius controls how large the hollow opening is.
  // inner_half_height is made slightly larger than the outer height so the
  // hollow core cuts through the top and bottom.
  const double inner_radius      = 8.0;
  const double inner_half_height = 10.0;

  // Diffuse-interface width.
  // Do not make this too small. A value of 1.0 can be too sharp for the mesh
  // and may cause instability or disappearance.
  const double interface_width = 3.0;

  if (index == 0)
    {
      // Variable 0: u
      scalar_value = u0;
    }
  else if (index == 1)
    {
      // Variable 1: phi
      // phi = +1 is the ice/solid seed.
      // phi = -1 is the vapor/background.
      //
      // This IC creates:
      // outer hexagonal prism - inner hollow core.
      //
      // So the seed starts as a hollow prism shell instead of a solid blob.

      const auto &sizes =
        this->get_user_inputs().get_spatial_discretization().get_size();

      const double xc = center[0] * sizes[0];
      const double yc = (dim > 1) ? center[1] * sizes[1] : 0.0;
      const double zc = (dim > 2) ? center[2] * sizes[2] : 0.0;

      const double dx = point[0] - xc;
      const double dy = (dim > 1) ? point[1] - yc : 0.0;
      const double dz = (dim > 2) ? point[2] - zc : 0.0;

      // Hexagonal radius approximation in the x-y plane.
      // This creates a faceted hexagonal cross-section instead of a circle.
      const double abs_x = std::abs(dx);
      const double abs_y = std::abs(dy);

      const double hex_radius =
        std::max(abs_x, 0.5 * abs_x + (std::sqrt(3.0) / 2.0) * abs_y);

      // Signed-distance-like value for the outer hexagonal prism.
      // Negative means inside the outer prism.
      const double outer_distance =
        std::max(hex_radius - outer_radius, std::abs(dz) - outer_half_height);

      // Signed-distance-like value for the inner hollow core.
      // Negative means inside the hollow core.
      const double inner_distance =
        std::max(hex_radius - inner_radius, std::abs(dz) - inner_half_height);

      // Solid shell = inside outer prism AND outside inner core.
      //
      // signed_distance < 0  -> solid ice shell
      // signed_distance > 0  -> vapor/background or hollow core
      const double signed_distance = std::max(outer_distance, -inner_distance);

      scalar_value =
        -std::tanh(signed_distance /
                   (std::sqrt(2.0) * interface_width));
    }
  else if (index == 2)
    {
      // Variable 2: xi1
      scalar_value = 0.0;
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::set_nonuniform_dirichlet(
  [[maybe_unused]] const unsigned int       &index,
  [[maybe_unused]] const unsigned int       &boundary_id,
  [[maybe_unused]] const unsigned int       &component,
  [[maybe_unused]] const dealii::Point<dim> &point,
  [[maybe_unused]] number                   &scalar_value,
  [[maybe_unused]] number                   &vector_component_value) const
{}

#include "custom_pde.inst"

PRISMS_PF_END_NAMESPACE
