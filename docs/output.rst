Output
======

The `Output` class processes the solution to provide output, which can be in the form of a dictionary of arrays, Pandas dataframes, or an Excel file. The dictionary keys (or sheet names in the case of Excel output) provide a complete output of quantities.

Gas species
-----------

Species output have a dictionary key associated with the species name and its state of aggregation (e.g., CO2_g, H2_g).

All gas species
~~~~~~~~~~~~~~~

.. list-table:: Outputs for gas species
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - gas_mass
     - kg
     - Mass in the gas
   * - gas_number
     - mol
     - Number of moles in the gas
   * - gas_number_density
     - mol m\ :math:`^{-3}`
     - Number density in the gas
   * - dissolved_mass
     - kg
     - Mass dissolved in the melt
   * - dissolved_number
     - mol
     - Number of moles in the melt
   * - dissolved_number_density
     - mol m\ :math:`^{-3}`
     - Number density in the melt
   * - dissolved_ppmw
     - kg kg\ :math:`^{-1}` (ppm by weight)
     - Dissolved mass relative to melt mass
   * - fugacity
     - bar
     - Fugacity
   * - fugacity_coefficient
     - dimensionless
     - Fugacity relative to (partial) pressure
   * - molar_mass
     - kg mol\ :math:`^{-1}`
     - Molar mass
   * - pressure
     - bar
     - Partial pressure
   * - total_mass
     - kg
     - Mass in all reservoirs
   * - total_number
     - mol
     - Number of moles in all reservoirs
   * - total_number_density
     - mol m\ :math:`^{-3}`
     - Number density in all reservoirs
   * - volume_mixing_ratio
     - mol mol\ :math:`^{-1}`
     - Volume mixing ratio in the gas
   * - gas_mass_fraction
     - kg kg\ :math:`^{-1}`
     - Mass fraction in the gas

O2_g additional outputs
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Additional outputs for O2_g
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - log10dIW_1_bar
     - dimensionless
     - Log10 shift relative to the IW buffer at 1 bar
   * - log10dIW_P
     - dimensionless
     - Log10 shift relative to the IW buffer at the total pressure

Condensed species
-----------------

Species output have a dictionary key associated with the species name and its state of aggregation (e.g., H2O_l, S_cr).

.. list-table:: Outputs for condensed species
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - activity
     - dimensionless
     - Activity
   * - molar_mass
     - kg mol\ :math:`^{-1}`
     - Molar mass
   * - total_mass
     - kg
     - Mass
   * - total_number
     - mol
     - Number of moles
   * - total_number_density
     - mol m\ :math:`^{-3}`
     - Number density

Elements
--------

Element outputs have a dictionary key associated with the element name with an `element_` prefix (e.g., element_H, element_S).

.. list-table:: Outputs for elements
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - gas_mass
     - kg
     - Mass in the gas
   * - gas_number
     - mol
     - Number of moles in the gas
   * - gas_number_density
     - mol m\ :math:`^{-3}`
     - Number density in the gas
   * - condensed_mass
     - kg
     - Mass in condensed species
   * - condensed_number
     - mol
     - Number of moles in condensed species
   * - condensed_number_density
     - mol m\ :math:`^{-3}`
     - Number density in condensed species
   * - degree_of_condensation
     - dimensionless
     - Degree of condensation
   * - dissolved_mass
     - kg
     - Mass dissolved in the melt
   * - dissolved_number
     - mol
     - Number of moles in the melt
   * - dissolved_number_density
     - mol m\ :math:`^{-3}`
     - Number density in the melt
   * - logarithmic_abundance
     - dimensionless
     - Logarithmic abundance
   * - molar_mass
     - kg mol\ :math:`^{-1}`
     - Molar mass
   * - total_mass
     - kg
     - Mass in all reservoirs
   * - total_number
     - mol
     - Number of moles in all reservoirs
   * - total_number_density
     - mol m\ :math:`^{-3}`
     - Number density in all reservoirs
   * - volume_mixing_ratio
     - mol mol\ :math:`^{-1}`
     - Volume mixing ratio

Thermodynamic system
--------------------

The thermodynamic system output has a dictionary key of `system`. The exact set of outputs depends on the type of thermodynamic system being considered:

.. list-table:: Outputs for all thermodynamic systems
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - temperature
     - K
     - Temperature
   * - pressure
     - bar
     - Pressure   
   * - volume
     - m\ :math:`^3`
     - Volume

For a planet, the thermodynamic system provides the following additional outputs:

.. list-table:: Planet-specific outputs
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - core_mass_fraction
     - kg kg\ :math:`^{-1}`
     - Mass fraction of iron core relative to total planet mass
   * - mantle_mass
     - kg
     - Mass of the silicate mantle
   * - mantle_melt_fraction
     - kg kg\ :math:`^{-1}`
     - Fraction of silicate mantle that is molten
   * - mantle_melt_mass
     - kg
     - Mass of molten silicate
   * - mantle_solid_mass
     - kg
     - Mass of solid silicate
   * - planet_mass
     - kg
     - Total mass of the planet
   * - surface_area
     - m\ :math:`^2`
     - Surface area at the surface radius
   * - surface_gravity
     - m s\ :math:`^{-2}`
     - Gravitational acceleration at the surface radius
   * - surface_radius
     - m
     - Radius of the planetary surface

Gas phase (totals)
------------------

The gas phase output has a dictionary key of `gas`.

.. list-table:: Outputs for gas
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - species_number
     - mol
     - Number of moles of species
   * - species_number_density
     - mol m\ :math:`^{-3}`
     - Number density of species
   * - mass
     - kg
     - Mass
   * - molar_mass
     - kg mol\ :math:`^{-1}`
     - Molar mass
   * - element_number
     - mol
     - Number of moles of elements
   * - element_number_density
     - mol m\ :math:`^{-3}`
     - Number density of elements
  
Other output
------------

- constraints: Applied elemental mass and/or species fugacity constraints
- raw: Raw solution from the solver, i.e. number of moles and stabilities
- residual: Residuals of the reaction network and mass balance
- solver: Solver quantities