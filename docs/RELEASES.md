# Library Name Release Notes

## 1.0.12 - 2024-04-16

### Added
- corkit/lasco.py (CME) class (Plot)
    - CME.mass(bn, fn): Given a base image, compute CME mass of fn file.
    - CME.plot(mass, fn_header): Plots CME.
- README.md (Citation)
    - This library can now be cited.
- docs/RELEASES.md
    - Historic record for each release.
- corkit/lasco.py (Plot) class (object)
    - imshow(img, header): Visualization of coronagraph data and metadata.

### Changed
- None

### Deprecated
- corkit/reconstruction.py
    - image_reconstruction (function)
    - fuzzy_image (function)
    - read_zone (function)
    - dct (function)
    - fuzzy_block (function)
    - num_to_fuzzy (function)
    - inter_fuzzy (function)
    - fuzzy_to_num (function)
    - read_block (function)
    - getl05hdrparam (function)

### Removed
- None

### Fixed
- corkit/lasco.py
    - c2_calibrate (function): None type error on calibration forward, bias not defined for some files.
    - c3_calibrate (function): None type error on calibration forward, bias not defined for some files.

### Security
- None

### Contributors
- [Jorge Enciso](https://github.com/Jorgedavyd)

---

## 1.1.0 - 2024-04-21

### Added
- corkit/secchi.py (level_05, level_1) tuple[function, function]
    - level_05: level 0 -> level 0.5 routine.
    - level_1: level 0.5 -> level 1 routine.
- corkit/reconstruction.py (CoronagraphReconstruction) class (nn.Module)
    - Partial convolutions based model for missing block inpainting.
### Changed
- 

### Deprecated
- None 

### Removed
- corkit/reconstruction.py
    - image_reconstruction (function)
    - fuzzy_image (function)
    - read_zone (function)
    - dct (function)
    - fuzzy_block (function)
    - num_to_fuzzy (function)
    - inter_fuzzy (function)
    - fuzzy_to_num (function)
    - read_block (function)
    - getl05hdrparam (function)

### Fixed
- 

### Security
- 

### Contributors
- [Jorge Enciso](https://github.com/Jorgedavyd)
