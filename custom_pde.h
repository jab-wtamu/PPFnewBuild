#pragma once

#include <prismspf/core/pde_operator.h>
#include <prismspf/core/phase_field_tools.h>
#include <prismspf/core/variable_attribute_loader.h>
#include <prismspf/core/variable_attributes.h>

#include <prismspf/user_inputs/user_input_parameters.h>

#include <prismspf/utilities/utilities.h>

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE

class CustomAttributeLoader : public VariableAttributeLoader
{
public:
  ~CustomAttributeLoader() override = default;

  void
  load_variable_attributes() override;
};

template <unsigned int dim, unsigned int degree, typename number>
class CustomPDE : public PDEOperator<dim, degree, number>
{
public:
  using ScalarValue = dealii::VectorizedArray<number>;
  using ScalarGrad  = dealii::Tensor<1, dim, dealii::VectorizedArray<number>>;
  using ScalarHess  = dealii::Tensor<2, dim, dealii::VectorizedArray<number>>;
  using VectorValue = dealii::Tensor<1, dim, dealii::VectorizedArray<number>>;
  using VectorGrad  = dealii::Tensor<2, dim, dealii::VectorizedArray<number>>;
  using VectorHess  = dealii::Tensor<3, dim, dealii::VectorizedArray<number>>;

  explicit CustomPDE(const UserInputParameters<dim> &_user_inputs,

                     PhaseFieldTools<dim>           &_pf_tools)

    : PDEOperator<dim, degree, number>(_user_inputs, _pf_tools)

  {}

private:
  void
  set_initial_condition(const unsigned int       &index,
                        const unsigned int       &component,
                        const dealii::Point<dim> &point,
                        number                   &scalar_value,
                        number                   &vector_component_value) const override;

  void
  set_nonuniform_dirichlet(const unsigned int       &index,
                           const unsigned int       &boundary_id,
                           const unsigned int       &component,
                           const dealii::Point<dim> &point,
                           number                   &scalar_value,
                           number                   &vector_component_value) const override;

  void
  compute_explicit_rhs(
    VariableContainer<dim, degree, number>                    &variable_list,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
    const dealii::VectorizedArray<number>                     &element_volume,
    Types::Index                                               solve_block) const override;

  void
  compute_nonexplicit_rhs(
    VariableContainer<dim, degree, number>                    &variable_list,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
    const dealii::VectorizedArray<number>                     &element_volume,
    Types::Index                                               solve_block,
    Types::Index current_index = Numbers::invalid_index) const override;

  void
  compute_nonexplicit_lhs(
    VariableContainer<dim, degree, number>                    &variable_list,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
    const dealii::VectorizedArray<number>                     &element_volume,
    Types::Index                                               solve_block,
    Types::Index current_index = Numbers::invalid_index) const override;

  void
  compute_postprocess_explicit_rhs(
    VariableContainer<dim, degree, number>                    &variable_list,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
    const dealii::VectorizedArray<number>                     &element_volume,
    Types::Index                                               solve_block) const override;

  // ================================================================
  // Snow / Demange model constants
  // ================================================================

  number u0 =
    this->get_user_inputs().get_user_constants().get_model_constant_double("u0");

  number eps_xy =
    this->get_user_inputs().get_user_constants().get_model_constant_double("eps_xy");

  number eps_z =
    this->get_user_inputs().get_user_constants().get_model_constant_double("eps_z");

  number Gamma =
    this->get_user_inputs().get_user_constants().get_model_constant_double("Gamma");

  number lambda =
    this->get_user_inputs().get_user_constants().get_model_constant_double("lambda");

  number D_tilde =
    this->get_user_inputs().get_user_constants().get_model_constant_double("D_tilde");

  number Lsat =
    this->get_user_inputs().get_user_constants().get_model_constant_double("Lsat");

  number regval =
    this->get_user_inputs().get_user_constants().get_model_constant_double("regval");
};

PRISMS_PF_END_NAMESPACE
