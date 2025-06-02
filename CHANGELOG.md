
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [2.0.0] - 2024-06-07
 
New structure of folder and documentation.
 
### Added
 
### Changed
 
### Fixed

## [2.0.1] - 2024-06-12

### Added
wrapper_grid_calc_sozh.py
    sensitivity study for different physical properties

### Changed
test_script_multiprocess.py
    methods to update physical properties properties

### Fixed
 separator_model.py
    methods to update physical properties and operating conditions from tuple

## [2.1.0] - 2025-03-10
### Added
scripts/wrapper_calc_flooding_sozh.py
    wrapper function to calculate flooding points

scripts/test_script_searchflooding_sozh.py
    test script for wrapper function to calculate flooding points

scripts/extract_results.py
    script to extract flooding point data from wrapper loops
### Changed
moved supporting scripts to subfolder "scripts" copy paste scripts from here to main folder for usage

separator_model.py
    added functions to calculate the flooding point (volume flow entering the separator until a steady state dpz height is achieved at const. h_w)
    modified function "solve_ODE" to account for const. h_w and/or const. h_l by adjusting the outflows
### Fixed

## [2.2.0] - 2025-05-05
### Added
physical_prop_t-dependent_1octanol.csv file added with temperature dependent physical properties of 1-octanol in water

### Changed

### Fixed
separator_model.py
    modified function "solve_ODE" to account for const. h_w and/or const. h_l by adjusting the outflows


