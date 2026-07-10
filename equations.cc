#include "custom_pde.h"

#include <prismspf/core/type_enums.h>
#include <prismspf/core/variable_attribute_loader.h>
#include <prismspf/core/variable_container.h>

#include <prismspf/config.h>

#include <cmath>
#include <algorithm>
#include <vector>

PRISMS_PF_BEGIN_NAMESPACE

void
CustomAttributeLoader::load_variable_attributes()
{
  // ============================================================
  // VARIABLE REGISTRATION / EQUATION-DEPENDENCY SETUP
  // ============================================================
  // This section does not solve the PDEs themselves.
  // It tells PRISMS-PF:
  //   - what variables exist,
  //   - what type they are,
  //   - whether they are time-dependent or auxiliary,
  //   - which quantities each equation depends on.

  // ------------------------------------------------------------
  // Variable 0: u
  // Physical meaning:
  //   u = reduced supersaturation field.
  // Governing equation from the paper:
  //   ∂t u = D̃ ∇Γ·( q(ϕ) ∇Γ u ) - (Lsat/2) B(n) ∂tϕ
  // Here it is treated as an explicit time-dependent scalar field.
  // ------------------------------------------------------------
  set_variable_name(0, "u");
  set_variable_type(0, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(0, ExplicitTimeDependent);
  // RHS value-term dependencies for u equation:
  //   depends on u, xi1, phi, grad(phi), grad(u)
  //   because the explicit update later uses u, xi1, B(n), A(n), q(phi), and flux terms.
  set_dependencies_value_term_rhs(0, "u,xi1,phi,grad(phi),grad(u)");
  // RHS gradient-term dependencies for u equation:
  //   diffusion flux depends on phi and grad(u)
  set_dependencies_gradient_term_rhs(0, "phi,grad(u)");

  // ------------------------------------------------------------
  // Variable 1: phi
  // Physical meaning:
  //   phi = phase-field variable/order parameter.
  // Governing equation from the paper:
  //   A(n)^2 ∂tϕ = f'(ϕ) + λ B(n) g'(ϕ) u
  //                + (1/2) ∇Γ·[ |∇ϕ|^2 ∂(A(n)^2)/∂(∇ϕ) + A(n)^2 ∇Γϕ ]
  // In this implementation, phi is updated explicitly using xi1 / A(n)^2,
  // where xi1 stores the non-time-derivative RHS contribution.
  // ------------------------------------------------------------
  set_variable_name(1, "phi");
  set_variable_type(1, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(1, ExplicitTimeDependent);
  // RHS value-term dependencies for phi equation:
  //   depends on phi, xi1, grad(phi)
  set_dependencies_value_term_rhs(1, "phi,xi1,grad(phi)");
  // No separate gradient term is assigned directly for phi here.
  set_dependencies_gradient_term_rhs(1, "");

  // ------------------------------------------------------------
  // Variable 2: xi1
  // Physical meaning in this code:
  //   xi1 is an auxiliary field used to store the non-explicit RHS of the phi equation.
  // It represents the quantity:
  //   xi1 = -f'(phi) + λ B(n) g'(phi) u + anisotropic surface-tension operator
  // so that later the explicit phi update is:
  //   phi^{n+1} = phi^n + Δt * xi1 / A(n)^2
  // ------------------------------------------------------------
  set_variable_name(2, "xi1");
  set_variable_type(2, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(2, Auxiliary);
  // xi1 depends on phi, u, grad(phi)
  set_dependencies_value_term_rhs(2, "phi,u,grad(phi)");
  set_dependencies_gradient_term_rhs(2, "grad(phi)");
}

namespace
{
  constexpr double snow_pi = 3.14159265358979323846;

  inline double
  map_theta_to_principal_sector(const double theta_raw)
  {
    // ============================================================
    // THETA SECTOR MAPPING
    // ============================================================
    // Snow crystal horizontal anisotropy has 6-fold symmetry:
    //   cos(6 theta)
    // Therefore theta can be mapped into one principal sector of width pi/3.
    // This reduces the angle into the canonical range centered around 0,
    // specifically approximately [-pi/6, pi/6].
    double theta_local = std::remainder(theta_raw, snow_pi / 3.0);

    if (theta_local <= -snow_pi / 6.0)
      theta_local += snow_pi / 3.0;
    else if (theta_local > snow_pi / 6.0)
      theta_local -= snow_pi / 3.0;

    return theta_local;
  }

  inline double
  map_psi_to_principal_sector(const double psi_raw)
  {
    // ============================================================
    // PSI SECTOR MAPPING
    // ============================================================
    // Vertical anisotropy uses:
    //   cos(2 psi)
    // which is symmetric over pi.
    // This maps psi into the principal interval [0, pi/2].
    double psi_wrap = std::fmod(psi_raw, snow_pi);
    if (psi_wrap < 0.0)
      psi_wrap += snow_pi;

    return (psi_wrap <= snow_pi / 2.0) ? psi_wrap : (snow_pi - psi_wrap);
  }

  inline double
  raw_snow_A(const double theta_local,
             const double psi_local,
             const double eps_xy,
             const double eps_z)
  {
    // ============================================================
    // RAW ANISOTROPY FUNCTION A(n)
    // ============================================================
    // Main paper anisotropy formula:
    //   A(n) = 1 + eps_xy cos(6 theta) + eps_z cos(2 psi)
    // This is the unregularized / direct formula.
    return 1.0 + eps_xy * std::cos(6.0 * theta_local) +
           eps_z * std::cos(2.0 * psi_local);
  }

  // ============================================================
  // ORIGINAL PYTHON-TO-C++ TABLE IMPLEMENTATION
  // ============================================================
  // This block keeps the same lookup-table method from the standalone
  // Python-to-C++ conversion.
  //
  // Original standalone purpose:
  //   - Build theta_table and phi_table.
  //   - Compute phi_max_table and theta_max_table.
  //   - Use interpolation to estimate phi_m and theta_m.
  //
  // Integration change for PRISMS-PF:
  //   - The timing/printing/test arrays were removed because they do not belong
  //     inside equation.cc.
  //   - epsxy and epsz are passed in from the PRISMS-PF material constants
  //     instead of being hard-coded as global standalone variables.
  //   - The table is built once on first use, then reused by interpolation.
  constexpr int missing_angle_table_size = 100000;
  constexpr double missing_angle_eps_floor = 1.0e-14;
  constexpr double anisotropy_division_floor = 1.0e-12;
  constexpr double angle_floor = 1.0e-8;

  // Keep value inside a safe range.
  inline double
  clamp_value(double value, double min_value, double max_value)
  {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
  }

  /*
      Same f1 equation as the standalone Python-to-C++ table code.

      The unknown is x, where:

          x = cos(2*theta_m)

      phi is passed directly because C++ does not use the same outside-scope
      behavior as the Python function.
  */
  inline double
  f1(double x, double phi, double epsxy, double epsz)
  {
    return epsxy * (20.0*x*x*x + 24.0*x*x - 3.0*x - 6.0)
           - (1.0 + epsz * std::cos(2.0 * phi));
  }

  // Derivative of f1 with respect to x.
  inline double
  f1_prime(double x, double epsxy)
  {
    return epsxy * (60.0*x*x + 48.0*x - 3.0);
  }

  /*
      Replacement for scipy.optimize.root_scalar().

      This solves:

          f1(x, phi) = 0

      where:

          x = cos(2*theta_m)

      The valid range is:

          -1 <= x <= 1

      Newton is used when safe. Bisection is used when Newton would leave
      the valid bracket.
  */
  inline double
  root_scalar(double phi, double x0, double epsxy, double epsz)
  {
    if (std::abs(epsxy) < missing_angle_eps_floor)
      return 2.0;

    double left  = -1.0;
    double right =  1.0;

    double f_left = f1(left, phi, epsxy, epsz);
    double f_right = f1(right, phi, epsxy, epsz);

    // Check if an endpoint is already the root.
    if (std::abs(f_left) < 1.0e-14)
      return left;

    if (std::abs(f_right) < 1.0e-14)
      return right;

    // Make sure the root is bracketed.
    // If no valid root is found, return an invalid x value.
    // Valid x = cos(2*theta_m) must be inside [-1, 1].
    // Returning 2.0 prevents the caller from treating this as a valid root.
    if (f_left * f_right > 0.0)
      return 2.0;

    // Start from the initial guess.
    double x = clamp_value(x0, left, right);

    const int max_iter = 100;
    const double tolerance = 1.0e-14;

    for (int iter = 0; iter < max_iter; ++iter)
      {
        double fx = f1(x, phi, epsxy, epsz);

        // Stop if f(x) is close to zero.
        if (std::abs(fx) < tolerance)
          return x;

        // Update the bracket.
        if (f_left * fx <= 0.0)
          {
            right = x;
          }
        else
          {
            left = x;
            f_left = fx;
          }

        double dfx = f1_prime(x, epsxy);
        double x_new;

        // Try Newton's method.
        if (std::abs(dfx) > 1.0e-14)
          {
            x_new = x - fx / dfx;

            // Fall back to bisection if Newton is unsafe.
            if (!std::isfinite(x_new) || x_new <= left || x_new >= right)
              x_new = 0.5 * (left + right);
          }
        else
          {
            // Use bisection if derivative is too small.
            x_new = 0.5 * (left + right);
          }

        // Stop if the result is no longer changing much.
        if (std::abs(x_new - x) < tolerance)
          return x_new;

        x = x_new;
      }

    return x;
  }

  /*
      Replacement for np.interp().

      This performs linear interpolation between neighboring table values.
  */
  inline double
  interp(double x_query,
         const std::vector<double>& x_table,
         const std::vector<double>& y_table)
  {
    if (x_query <= x_table.front())
      return y_table.front();

    if (x_query >= x_table.back())
      return y_table.back();

    // Find the first table value greater than or equal to x_query.
    auto upper = std::lower_bound(x_table.begin(), x_table.end(), x_query);

    int i1 = static_cast<int>(upper - x_table.begin());
    int i0 = i1 - 1;

    double x0 = x_table[i0];
    double x1 = x_table[i1];

    double y0 = y_table[i0];
    double y1 = y_table[i1];

    double weight = (x_query - x0) / (x1 - x0);

    return y0 + weight * (y1 - y0);
  }

  struct MissingOrientationTables
  {
    std::vector<double> theta_table;
    std::vector<double> phi_max_table;
    std::vector<double> phi_table;
    std::vector<double> theta_max_table;

    MissingOrientationTables(const double epsxy, const double epsz)
      : theta_table(missing_angle_table_size),
        phi_max_table(missing_angle_table_size, 0.0),
        phi_table(missing_angle_table_size),
        theta_max_table(missing_angle_table_size, 0.0)
    {
      /*
          Python equivalent:

              theta_table = np.linspace(-pi/6, pi/6, N)
              phi_max_table = theta_table * 0.0
      */
      for (int i = 0; i < missing_angle_table_size; ++i)
        {
          double ratio = static_cast<double>(i) /
                         static_cast<double>(missing_angle_table_size - 1);

          theta_table[i] = -snow_pi / 6.0 + ratio * (snow_pi / 3.0);

          if (std::abs(epsz) > missing_angle_eps_floor)
            {
              double sol =
                (1.0 + epsxy * std::cos(6.0 * theta_table[i])) / epsz - 2.0;

              if (std::abs(sol) <= 1.0)
                {
                  phi_max_table[i] =
                    0.5 * std::acos(clamp_value(sol, -1.0, 1.0));
                }
              else
                {
                  phi_max_table[i] = 0.0;
                }
            }
          else
            {
              phi_max_table[i] = 0.0;
            }
        }

      /*
          Python equivalent:

              phi_table = np.linspace(0, pi, N)
              theta_max_table = phi_table * 0.0
      */
      for (int i = 0; i < missing_angle_table_size; ++i)
        {
          double ratio = static_cast<double>(i) /
                         static_cast<double>(missing_angle_table_size - 1);

          phi_table[i] = ratio * snow_pi;

          double phi = phi_table[i];

          /*
              This follows the Python-to-C++ table structure.

              root_scalar solves for x, not theta_m directly:

                  x = cos(2*theta_m)
          */
          if (std::abs(epsxy) > missing_angle_eps_floor)
            {
              double sol_root = root_scalar(phi, snow_pi / 12.0, epsxy, epsz);

              if (std::abs(sol_root) <= 1.0)
                {
                  theta_max_table[i] =
                    0.5 * std::acos(clamp_value(sol_root, -1.0, 1.0));
                }
              else
                {
                  theta_max_table[i] = 0.0;
                }
            }
          else
            {
              theta_max_table[i] = 0.0;
            }
        }
    }

    // Original table meaning:
    //   theta_max_table = theta_m(phi)
    inline double
    lookup_theta_m(const double phi_query) const
    {
      return interp(phi_query, phi_table, theta_max_table);
    }

    // Original table meaning:
    //   phi_max_table = phi_m(theta)
    inline double
    lookup_phi_m(const double theta_query) const
    {
      return interp(theta_query, theta_table, phi_max_table);
    }
  };

  inline const MissingOrientationTables&
  get_missing_orientation_tables(const double epsxy, const double epsz)
  {
    // Built once on first use, then reused for the rest of the run.
    // This assumes epsxy and epsz are fixed material parameters.
    static const MissingOrientationTables tables(epsxy, epsz);
    return tables;
  }

  inline double
  eval_regularized_snow_A(const double theta_raw,
                          const double psi_raw,
                          const double eps_xy,
                          const double eps_z)
  {
    // ============================================================
    // REGULARIZED ANISOTROPY EVALUATION
    // ============================================================
    // This routine evaluates the anisotropy A(n) with the supplementary
    // missing-orientation regularization.
    //
    // Overall logic:
    //   1. Map raw angles into principal symmetry sectors.
    //   2. Compute raw anisotropy A_raw.
    //   3. Compute critical anisotropy thresholds eps_xy_m and eps_z_m.
    //   4. Detect whether theta and/or psi missing orientations occur.
    //   5. If needed, look up theta_m and/or psi_m from precomputed tables.
    //   6. Build regularized branch formulas A_theta and/or A_psi.
    //   7. Return the correct piecewise value depending on angular region.
    const double small = 1.0e-12;

    const double theta_local = map_theta_to_principal_sector(theta_raw);
    const double psi_local   = map_psi_to_principal_sector(psi_raw);
    const double theta_abs   = std::abs(theta_local);

    // Raw anisotropy:
    //   A_raw = 1 + eps_xy cos(6 theta) + eps_z cos(2 psi)
    const double A_raw = raw_snow_A(theta_local, psi_local, eps_xy, eps_z);

    // Critical anisotropy thresholds from regularization logic.
    const double eps_xy_m = (1.0 + eps_z * std::cos(2.0 * psi_local)) / 35.0;
    const double eps_z_m  = (1.0 + eps_xy * std::cos(6.0 * theta_local)) / 3.0;

    // Missing-orientation detection.
    const bool theta_missing = (eps_xy > eps_xy_m);
    const bool psi_missing   = (eps_z > eps_z_m);

    // Get the original Python-to-C++ lookup tables.
    // They are built once on first use and then reused by interpolation.
    const MissingOrientationTables &missing_tables =
      get_missing_orientation_tables(eps_xy, eps_z);

    double theta_m = 0.0;
    double psi_m   = 0.0;

    if (theta_missing)
      // Original table meaning:
      //   theta_m = theta_max_table interpolated as a function of phi/psi.
      theta_m = missing_tables.lookup_theta_m(psi_local);

    if (psi_missing)
      // Original table meaning:
      //   psi_m here corresponds to the PDF's phi_m.
      //   It is phi_max_table interpolated as a function of |theta|.
      psi_m = missing_tables.lookup_phi_m(theta_abs);

    // ------------------------------------------------------------
    // Build theta-regularized branch A_theta
    // This corresponds to the supplement piece used when
    // |theta| < theta_m and psi is outside the psi-missing region.
    // General form:
    //   A^theta(theta,psi) = A1(psi) + B1(psi) cos(theta)
    // ------------------------------------------------------------
    double A_theta = A_raw;
    if (theta_missing && theta_m > 0.0)
      {
        const double B1 =
          6.0 * eps_xy * std::sin(6.0 * theta_m) / (std::sin(theta_m) + small);

        const double A1 =
          1.0 + eps_xy * std::cos(6.0 * theta_m) +
          eps_z * std::cos(2.0 * psi_local) - B1 * std::cos(theta_m);

        A_theta = A1 + B1 * std::cos(theta_local);
      }

    // ------------------------------------------------------------
    // Build psi-regularized branch A_psi
    // This corresponds to the supplement piece used when
    // psi < psi_m and |theta| is outside the theta-missing region.
    // General form:
    //   A^psi(theta,psi) = A2(theta) + B2(theta) cos(psi)
    // ------------------------------------------------------------
    double A_psi = A_raw;
    if (psi_missing && psi_m > 0.0)
      {
        const double B2 =
          2.0 * eps_z * std::sin(2.0 * psi_m) / (std::sin(psi_m) + small);

        const double A2 =
          1.0 + eps_xy * std::cos(6.0 * theta_local) +
          eps_z * std::cos(2.0 * psi_m) - B2 * std::cos(psi_m);

        A_psi = A2 + B2 * std::cos(psi_local);
      }

    // ------------------------------------------------------------
    // Piecewise return logic for regularized A(n)
    // Cases:
    //   1. both theta and psi missing
    //   2. only theta missing
    //   3. only psi missing
    //   4. no missing orientations -> return A_raw
    // ------------------------------------------------------------
    if (theta_missing && psi_missing && theta_m > 0.0 && psi_m > 0.0)
      {
        // Region 1:
        //   |theta| < theta_m, psi >= psi_m
        //   use theta-regularized branch
        if (theta_abs < theta_m && psi_local >= psi_m)
          return A_theta;
        // Region 2:
        //   |theta| >= theta_m, psi < psi_m
        //   use psi-regularized branch
        else if (theta_abs >= theta_m && psi_local < psi_m)
          return A_psi;
        // Region 3:
        //   |theta| < theta_m, psi < psi_m
        //   mixed/blended region between A_theta and A_psi
        else if (theta_abs < theta_m && psi_local < psi_m)
          {
            const double dtheta = theta_abs - theta_m;
            const double dpsi   = psi_local - psi_m;
            const double denom  = std::sqrt(dtheta * dtheta + dpsi * dpsi) + small;
            const double alpha  = std::abs(dtheta) / denom;

            // Blend:
            //   A = alpha A_theta + (1-alpha) A_psi
            return alpha * A_theta + (1.0 - alpha) * A_psi;
          }
        // Region 4:
        //   outside missing-orientation sectors -> raw formula
        else
          return A_raw;
      }
    else if (theta_missing && theta_m > 0.0)
      // Only theta missing:
      //   use A_theta for |theta| < theta_m, else A_raw
      return (theta_abs < theta_m) ? A_theta : A_raw;
    else if (psi_missing && psi_m > 0.0)
      // Only psi missing:
      //   use A_psi for psi < psi_m, else A_raw
      return (psi_local < psi_m) ? A_psi : A_raw;

    // No regularization needed.
    return A_raw;
  }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number>                    &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number>                     &element_volume,
  [[maybe_unused]] Types::Index                                               solve_block) const
{
  // ============================================================
  // EXPLICIT RHS ASSEMBLY
  // ============================================================
  // This function builds the explicit time-update terms for:
  //   - u (variable 0)
  //   - phi (variable 1)
  //
  // Paper equations being used here:
  //
  //   (Eq. 1) A(n)^2 ∂tϕ = [all non-time-derivative terms]
  //   (Eq. 2) ∂t u = D̃ ∇Γ·( q(ϕ) ∇Γ u ) - (Lsat/2) B(n) ∂tϕ
  //
  // In this implementation:
  //   xi1 stores the RHS of Eq. (1) excluding A(n)^2 ∂tϕ,
  // so the explicit phi update is:
  //   ∂t phi = xi1 / A2_n
  // and therefore:
  //   phi^{n+1} = phi + Δt * xi1 / A2_n
  //
  // For u, the explicit form assembled is:
  //   u^{n+1} = u - Δt * (Lsat/2) * (B_n/A2_n) * xi1
  // plus a gradient flux term for diffusion.

  // ------------------------------------------------------------
  // Read current field values and gradients
  // ------------------------------------------------------------
  ScalarValue u  = variable_list.template get_value<ScalarValue>(0);
  ScalarGrad  ux = variable_list.template get_gradient<ScalarGrad>(0);

  ScalarValue phi  = variable_list.template get_value<ScalarValue>(1);
  ScalarGrad  phix = variable_list.template get_gradient<ScalarGrad>(1);

  // xi1 is the auxiliary RHS for phi equation.
  ScalarValue xi1 = variable_list.template get_value<ScalarValue>(2);

  // ------------------------------------------------------------
  // Compute interface normal n = -grad(phi)/|grad(phi)|
  // ------------------------------------------------------------
  ScalarValue normgradn = std::sqrt(phix.norm_square());
  ScalarGrad  normal    = (-phix) / (normgradn + regval);

  // Quantities needed later:
  //   A2_n = A(n)^2
  //   A2_safe = A(n)^2 plus a tiny denominator floor, used only for division
  //   B_n  = kinetic anisotropy B(n)
  //   F2   = diffusion flux vector used in u equation gradient term
  ScalarValue A2_n;
  ScalarValue A2_safe;
  ScalarValue B_n;
  ScalarGrad  F2;

  if constexpr (dim == 3)
    {
      // --------------------------------------------------------
      // Extract components of unit normal n = (nx, ny, nz)
      // --------------------------------------------------------
      ScalarValue nx = normal[0];
      ScalarValue ny = normal[1];
      ScalarValue nz = normal[2];

      // rho_n = sqrt(nx^2 + ny^2)
      // used for the angular coordinate psi
      ScalarValue rho_n = std::sqrt(nx * nx + ny * ny);

      // theta = azimuthal angle in xy-plane
      // psi   = angle relative to z direction, as used in the model
      ScalarValue theta;
      ScalarValue psi;
      ScalarValue A_n;

      for (unsigned int i = 0; i < theta.size(); ++i)
        {
          // ----------------------------------------------------
          // ANGLE DEFINITIONS FROM NORMAL VECTOR
          // ----------------------------------------------------
          // theta = atan2(ny, nx)
          // psi   = atan2(sqrt(nx^2 + ny^2), nz)
          theta[i] = std::atan2(ny[i], nx[i]);
          psi[i]   = std::atan2(rho_n[i], nz[i]);

          // ----------------------------------------------------
          // Evaluate regularized anisotropy A(n)
          // ----------------------------------------------------
          A_n[i] = eval_regularized_snow_A(theta[i],
                                           psi[i],
                                           static_cast<double>(eps_xy),
                                           static_cast<double>(eps_z));
        }

      // A2_n = physical A(n)^2.
      // A2_safe is used only where A2 appears in a denominator.
      A2_n    = A_n * A_n;
      A2_safe = A2_n + anisotropy_division_floor;

      // --------------------------------------------------------
      // KINETIC ANISOTROPY B(n)
      // --------------------------------------------------------
      // From the paper:
      //   B(n) = sqrt(nx^2 + ny^2 + Gamma^2 nz^2)
      B_n = std::sqrt(nx * nx + ny * ny + Gamma * Gamma * nz * nz);

      // --------------------------------------------------------
      // q(phi) used in diffusion equation for u
      // --------------------------------------------------------
      // From the paper:
      //   q(phi) = 1 - phi
      ScalarValue q_phi = 1.0 - phi;

      // --------------------------------------------------------
      // DIFFUSION FLUX FOR u EQUATION (RHS gradient contribution)
      // --------------------------------------------------------
      // Paper diffusion operator:
      //   D̃ ∇Γ · ( q(phi) ∇Γ u )
      // Here the flux vector is assembled componentwise.
      // Since ∇Γ = (∂x, ∂y, Gamma ∂z), the z contribution picks up Gamma^2.
      //
      // F2 corresponds to:
      //   F2 = D̃ q(phi) [ ux, uy, Gamma^2 uz ]
      F2[0] = D_tilde * q_phi * ux[0];
      F2[1] = D_tilde * q_phi * ux[1];
      F2[2] = D_tilde * q_phi * Gamma * Gamma * ux[2];
    }
  else
    {
      // 2D fallback / simplified isotropic case
      ScalarValue q_phi = 1.0 - phi;
      A2_n    = 1.0;
      A2_safe = 1.0;
      B_n     = 1.0;
      F2      = D_tilde * q_phi * ux;
    }

  // ------------------------------------------------------------
  // EXPLICIT UPDATE FOR phi
  // ------------------------------------------------------------
  // LHS from paper:
  //   A(n)^2 ∂t phi
  // RHS represented here by xi1
  // Therefore:
  //   ∂t phi = xi1 / A2_safe
  // Time-discrete explicit update:
  //   eq_phi = phi + Δt * xi1 / A2_safe
  ScalarValue eq_phi = phi + this->get_timestep() * xi1 / A2_safe;

  // ------------------------------------------------------------
  // VALUE TERM FOR u EQUATION
  // ------------------------------------------------------------
  // From paper term:
  //   -(Lsat/2) B(n) ∂t phi
  // and since ∂t phi = xi1 / A2_safe,
  // this becomes:
  //   -(Lsat/2) * (B_n / A2_safe) * xi1
  // Explicit update:
  //   eq_u = u - Δt * (Lsat/2) * (B_n/A2_safe) * xi1
  ScalarValue eq_u =
    u - this->get_timestep() * (Lsat / static_cast<number>(2.0)) * (B_n / A2_safe) * xi1;

  // ------------------------------------------------------------
  // GRADIENT TERM FOR u EQUATION
  // ------------------------------------------------------------
  // Diffusion part assembled as a gradient contribution.
  // Since the solver typically handles gradients as divergence-form terms,
  // the sign is written as minus the flux:
  //   eqx_u = -Δt * F2
  ScalarGrad eqx_u = -this->get_timestep() * F2;

  // ------------------------------------------------------------
  // STORE EXPLICIT RHS CONTRIBUTIONS
  // ------------------------------------------------------------
  // Variable 0 (u):
  //   value term   = eq_u
  //   gradient term = eqx_u
  // Variable 1 (phi):
  //   value term   = eq_phi
  variable_list.set_value_term(0, eq_u);
  variable_list.set_gradient_term(0, eqx_u);
  variable_list.set_value_term(1, eq_phi);
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_nonexplicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number>                    &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number>                     &element_volume,
  [[maybe_unused]] Types::Index                                               solve_block,
  [[maybe_unused]] Types::Index                                               current_index) const
{
  // ============================================================
  // NON-EXPLICIT RHS ASSEMBLY
  // ============================================================
  // In this code, only variable 2 (xi1) is assembled here.
  // xi1 stores the non-time-derivative RHS of the phi equation.
  //
  // Paper Eq. (1):
  //   A(n)^2 ∂t phi = f'(phi) + λ B(n) g'(phi) u
  //                   + (1/2) ∇Γ · [ |∇phi|^2 ∂(A^2)/∂(∇phi) + A^2 ∇Γ phi ]
  //
  // This routine builds the RHS pieces corresponding to:
  //   xi1 = -f'(phi) + λ B(n) g'(phi) u + anisotropic divergence operator
  // so that later phi can be updated explicitly using xi1 / A(n)^2.
  if (current_index == 2)
    {
      // --------------------------------------------------------
      // Read current values needed for xi1 equation
      // --------------------------------------------------------
      ScalarValue u    = variable_list.template get_value<ScalarValue>(0);
      ScalarValue phi  = variable_list.template get_value<ScalarValue>(1);
      ScalarGrad  phix = variable_list.template get_gradient<ScalarGrad>(1);

      // Compute normal n = -grad(phi)/|grad(phi)|
      ScalarValue normgradn = std::sqrt(phix.norm_square());
      ScalarGrad  normal    = (-phix) / (normgradn + regval);

      // Quantities needed for xi1 assembly:
      //   A2_n = A(n)^2
      //   B_n  = kinetic anisotropy B(n)
      //   F1   = anisotropic surface-tension flux term
      ScalarValue A2_n;
      ScalarValue B_n;
      ScalarGrad  F1;

      if constexpr (dim == 3)
        {
          // ----------------------------------------------------
          // Extract normal-vector components
          // ----------------------------------------------------
          ScalarValue nx = normal[0];
          ScalarValue ny = normal[1];
          ScalarValue nz = normal[2];

          ScalarValue rho_n = std::sqrt(nx * nx + ny * ny);

          ScalarValue theta;
          ScalarValue psi;
          ScalarValue A_n;
          ScalarValue dA2_dtheta;
          ScalarValue dA2_dpsi;

          // Finite-difference step for numerical derivatives of A(n)
          // with respect to theta and psi.
          const number fd_step = static_cast<number>(1.0e-6);

          for (unsigned int i = 0; i < theta.size(); ++i)
            {
              // ------------------------------------------------
              // Convert interface normal into angular variables
              // ------------------------------------------------
              theta[i] = std::atan2(ny[i], nx[i]);
              psi[i]   = std::atan2(rho_n[i], nz[i]);

              // ------------------------------------------------
              // Evaluate A and finite-difference derivatives
              // ------------------------------------------------
              // A_here       = A(theta, psi)
              // A_theta_p/m  = A(theta +/- h, psi)
              // A_psi_p/m    = A(theta, psi +/- h)
              const double A_here =
                eval_regularized_snow_A(theta[i], psi[i],
                                        static_cast<double>(eps_xy),
                                        static_cast<double>(eps_z));
              const double A_theta_p =
                eval_regularized_snow_A(theta[i] + fd_step, psi[i],
                                        static_cast<double>(eps_xy),
                                        static_cast<double>(eps_z));
              const double A_theta_m =
                eval_regularized_snow_A(theta[i] - fd_step, psi[i],
                                        static_cast<double>(eps_xy),
                                        static_cast<double>(eps_z));
              const double A_psi_p =
                eval_regularized_snow_A(theta[i], psi[i] + fd_step,
                                        static_cast<double>(eps_xy),
                                        static_cast<double>(eps_z));
              const double A_psi_m =
                eval_regularized_snow_A(theta[i], psi[i] - fd_step,
                                        static_cast<double>(eps_xy),
                                        static_cast<double>(eps_z));

              // Numerical derivatives of the actual quantity used by the PDE:
              //   d(A^2)/dtheta ≈ [A(theta+h)^2 - A(theta-h)^2]/(2h)
              //   d(A^2)/dpsi   ≈ [A(psi+h)^2   - A(psi-h)^2]/(2h)
              //
              // This is equivalent to 2*A*dA/dangle for smooth A, but is
              // safer for the regularized, piecewise, table-interpolated A.
              A_n[i] = A_here;
              dA2_dtheta[i] =

                static_cast<number>(

                  (A_theta_p * A_theta_p - A_theta_m * A_theta_m) /

                  (2.0 * static_cast<double>(fd_step)));
              dA2_dpsi[i] =

                static_cast<number>(

                  (A_psi_p * A_psi_p - A_psi_m * A_psi_m) /

                  (2.0 * static_cast<double>(fd_step)));
            }

          // A2_n = physical A(n)^2. Do not add a denominator floor here,
          // because this value is used in the surface-tension flux.
          A2_n = A_n * A_n;

          // B(n) from the paper
          B_n = std::sqrt(nx * nx + ny * ny + Gamma * Gamma * nz * nz);

          // ----------------------------------------------------
          // Derivatives of angles wrt grad(phi)
          // ----------------------------------------------------
          // These are chain-rule ingredients to compute:
          //   d(A^2)/d(grad phi)
          ScalarValue gradxy2 =
            phix[0] * phix[0] + phix[1] * phix[1];
          ScalarValue grad2 =
            gradxy2 + phix[2] * phix[2];
          ScalarValue gradxy =
            std::sqrt(gradxy2);

          // The angular coordinates are singular when gradxy -> 0.
          // Use a separate angular floor rather than the phase-field normal
          // floor so derivative spikes are suppressed only near singular
          // angular coordinates.
          ScalarValue safe_gradxy =
            std::sqrt(gradxy2 + angle_floor * angle_floor);
          ScalarValue denom_theta =
            gradxy2 + angle_floor * angle_floor;
          ScalarValue denom_psi =
            grad2 + angle_floor * angle_floor;

          ScalarGrad dtheta_dgradphi;
          // dtheta/d(grad phi)
          dtheta_dgradphi[0] = -phix[1] / denom_theta;
          dtheta_dgradphi[1] =  phix[0] / denom_theta;
          dtheta_dgradphi[2] =  0.0;

          ScalarGrad dpsi_dgradphi;
          // dpsi/d(grad phi)
          dpsi_dgradphi[0] = (-phix[2] * phix[0]) / (safe_gradxy * denom_psi);
          dpsi_dgradphi[1] = (-phix[2] * phix[1]) / (safe_gradxy * denom_psi);
          dpsi_dgradphi[2] = gradxy / denom_psi;

          // ----------------------------------------------------
          // Chain rule for derivative of A^2 wrt grad(phi)
          // ----------------------------------------------------
          //   d(A^2)/d(grad phi)
          //     = d(A^2)/dtheta * dtheta/d(grad phi)
          //     + d(A^2)/dpsi   * dpsi/d(grad phi)
          ScalarGrad dA2_dgradphi =
            dA2_dtheta * dtheta_dgradphi + dA2_dpsi * dpsi_dgradphi;

          // |grad(phi)|^2
          ScalarValue gradphi2 = phix.norm_square();

          // ----------------------------------------------------
          // Anisotropic gradient operator ∇Γ phi
          // ----------------------------------------------------
          // From the paper:
          //   ∇Γ = (∂x, ∂y, Gamma ∂z)
          ScalarGrad Gamma_gradphi;
          Gamma_gradphi[0] = phix[0];
          Gamma_gradphi[1] = phix[1];
          Gamma_gradphi[2] = Gamma * phix[2];

          // ----------------------------------------------------
          // INSIDE OF THE SURFACE-TENSION DIVERGENCE TERM
          // ----------------------------------------------------
          // Paper term inside divergence:
          //   |grad(phi)|^2 d(A^2)/d(grad phi) + A^2 ∇Γ phi
          ScalarGrad inside_F1;
          inside_F1[0] = gradphi2 * dA2_dgradphi[0] + A2_n * Gamma_gradphi[0];
          inside_F1[1] = gradphi2 * dA2_dgradphi[1] + A2_n * Gamma_gradphi[1];
          inside_F1[2] = gradphi2 * dA2_dgradphi[2] + A2_n * Gamma_gradphi[2];

          // ----------------------------------------------------
          // F1 = 1/2 * anisotropic flux
          // ----------------------------------------------------
          // Corresponds to the flux whose divergence contributes to Eq. (1).
          // The z component gets an extra Gamma consistent with ∇Γ·(...)
          F1[0] = static_cast<number>(0.5) * inside_F1[0];
          F1[1] = static_cast<number>(0.5) * inside_F1[1];
          F1[2] = static_cast<number>(0.5) * Gamma * inside_F1[2];
        }
      else
        {
          // 2D fallback / simplified isotropic case
          A2_n = 1.0;
          B_n  = 1.0;
          F1   = 0.0 * phix;
        }

      // --------------------------------------------------------
      // LOCAL REACTION / COUPLING TERMS OF PHI EQUATION
      // --------------------------------------------------------
      // From paper:
      //   f(phi) = -phi^2/2 + phi^4/4
      // so:
      //   f'(phi) = -phi + phi^3
      // Hence:
      //   -f'(phi) = phi - phi^3
      ScalarValue minus_fprime = phi - phi * phi * phi;

      // From paper:
      //   g'(phi) = (1 - phi^2)^2
      ScalarValue gprime = (1.0 - phi * phi) * (1.0 - phi * phi);

      // --------------------------------------------------------
      // xi1 VALUE TERM = local RHS of phi equation
      // --------------------------------------------------------
      // Equation assembled here:
      //   eq_xi1 = -f'(phi) + lambda * B(n) * g'(phi) * u
      // This is the VALUE part (non-gradient part) of the phi RHS.
      ScalarValue eq_xi1 = minus_fprime + lambda * B_n * gprime * u;

      // --------------------------------------------------------
      // xi1 GRADIENT TERM = anisotropic surface-tension contribution
      // --------------------------------------------------------
      // Since F1 represents the flux inside the divergence term,
      // the solver receives the gradient contribution with a minus sign:
      //   eqx_xi1 = -F1
      ScalarGrad eqx_xi1 = -F1;

      // --------------------------------------------------------
      // STORE xi1 RHS CONTRIBUTIONS
      // --------------------------------------------------------
      // Variable 2 (xi1):
      //   value term    = eq_xi1
      //   gradient term = eqx_xi1
      variable_list.set_value_term(2, eq_xi1);
      variable_list.set_gradient_term(2, eqx_xi1);
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_nonexplicit_lhs(
  [[maybe_unused]] VariableContainer<dim, degree, number>                    &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number>                     &element_volume,
  [[maybe_unused]] Types::Index                                               solve_block,
  [[maybe_unused]] Types::Index                                               current_index) const
{
  // ============================================================
  // NON-EXPLICIT LHS
  // ============================================================
  // Empty in this implementation.
  // No additional non-explicit left-hand-side terms are assembled here.
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_postprocess_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number>                    &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number>                     &element_volume,
  [[maybe_unused]] Types::Index                                               solve_block) const
{
  // ============================================================
  // POSTPROCESS EXPLICIT RHS
  // ============================================================
  // Empty in this implementation.
  // No postprocessing RHS terms are assembled here.
}

#include "custom_pde.inst"

PRISMS_PF_END_NAMESPACE
