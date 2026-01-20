# h264/docs/SPEC_MAPPING.md
# H.264 Specification Mapping

Reference: ITU-T H.264 (03/2010) - Advanced video coding for generic audiovisual services

## Module → Spec Section Mapping

### bitstream/ ✅ Complete
| Function | Spec Section | Description |
|----------|--------------|-------------|
| `find_start_codes()` | Annex B | Byte stream format |
| `remove_emulation_prevention_bytes()` | 7.4.1 | NAL unit decoding |
| `parse_nal_header()` | 7.3.1 | NAL unit syntax |
| `extract_nal_units()` | Annex B | Full NAL extraction |
| `iter_nal_units()` | Annex B | Memory-efficient iteration |
| `NALUnitType` | Table 7-1 | NAL unit type enum |
| `NumpyBitReader.read_ue()` | 9.1 | Unsigned Exp-Golomb (pure NumPy) |
| `NumpyBitReader.read_se()` | 9.1.1 | Signed Exp-Golomb (pure NumPy) |
| `NumpyBitReader.read_te()` | 9.1.2 | Truncated Exp-Golomb (pure NumPy) |
| `BitReader.read_ue()` | 9.1 | Unsigned Exp-Golomb (bitstring) |
| `BitReader.read_se()` | 9.1.1 | Signed Exp-Golomb (bitstring) |
| `BitReader.read_te()` | 9.1.2 | Truncated Exp-Golomb (bitstring) |

### parameters/ ✅ Complete
| Function | Spec Section | Description |
|----------|--------------|-------------|
| `parse_sps()` | 7.3.2.1 | Sequence parameter set syntax |
| `SPS.profile_idc` | Table 7-3 | Profile indicator |
| `SPS.level_idc` | Annex A | Level indicator |
| `SPS.chroma_format_idc` | 7.4.2.1 | Chroma format (0-3) |
| `SPS VUI parsing` | Annex E | Video usability information |
| `parse_pps()` | 7.3.2.2 | Picture parameter set syntax |
| `PPS.entropy_coding_mode_flag` | 7.4.2.2 | CAVLC(0) vs CABAC(1) |
| `PPS.pic_init_qp_minus26` | 7.4.2.2 | Initial QP offset |

### slice/ ✅ Complete
| Function | Spec Section | Description |
|----------|--------------|-------------|
| `parse_slice_header()` | 7.3.3 | Slice header syntax |
| `SliceType` | Table 7-6 | Slice type values (I, P, B, SI, SP) |
| `SliceHeader.frame_num` | 7.4.3 | Frame number for reference management |
| `SliceHeader.pic_order_cnt_lsb` | 7.4.3 | POC least significant bits |
| `_parse_ref_pic_list_modification()` | 7.3.3.1 | Reference picture list modification |
| `_parse_dec_ref_pic_marking()` | 7.3.3.3 | Decoded reference picture marking |
| `SliceHeader.slice_qp_delta` | 7.4.3 | Per-slice QP adjustment |
| `SliceHeader.disable_deblocking_filter_idc` | 7.4.3 | Deblocking filter control |

### entropy/ ✅ Complete
| Function | Spec Section | Description |
|----------|--------------|-------------|
| `decode_coeff_token()` | 9.2.1 | CAVLC parsing - coeff_token |
| `decode_levels()` | 9.2.2 | CAVLC parsing - level values |
| `decode_total_zeros()` | 9.2.3 | CAVLC parsing - total zeros |
| `decode_run_before()` | 9.2.4 | CAVLC parsing - run_before |
| `decode_trailing_ones_signs()` | 9.2.1 | Sign bits for trailing ones |
| `decode_residual_block()` | 9.2 | Complete residual decoding |
| `calculate_nC()` | 9.2.1 | Context calculation |
| `COEFF_TOKEN_*` | Table 9-5 | coeff_token VLC tables |
| `TOTAL_ZEROS_*` | Table 9-7/9 | total_zeros VLC tables |
| `RUN_BEFORE` | Table 9-10 | run_before VLC tables |
| `ZIGZAG_4x4` | Figure 6-10 | 4x4 zigzag scan order |

### dequant/
| Function | Spec Section | Description |
|----------|--------------|-------------|
| `dequant_4x4()` | 8.5.11 | Scaling and transformation - dequant |
| `LEVEL_SCALE` | Table 8-13 | LevelScale lookup table |

### transform/
| Function | Spec Section | Description |
|----------|--------------|-------------|
| `idct_4x4()` | 8.5.12 | Inverse 4x4 transform |
| `hadamard_4x4()` | 8.5.12.1 | 4x4 luma DC Hadamard |
| `hadamard_2x2()` | 8.5.12.2 | 2x2 chroma DC Hadamard |

### intra/
| Function | Spec Section | Description |
|----------|--------------|-------------|
| `intra_16x16_vertical()` | 8.3.3.1 | Intra_16x16_Vertical |
| `intra_16x16_horizontal()` | 8.3.3.2 | Intra_16x16_Horizontal |
| `intra_16x16_dc()` | 8.3.3.3 | Intra_16x16_DC |
| `intra_16x16_plane()` | 8.3.3.4 | Intra_16x16_Plane |
| `intra_4x4_*()` | 8.3.1.2 | Intra_4x4 prediction modes |
| `intra_chroma_*()` | 8.3.4 | Intra chroma prediction |

### inter/ (Future)
| Function | Spec Section | Description |
|----------|--------------|-------------|
| `motion_compensation()` | 8.4.2 | Fractional sample interpolation |
| `mv_prediction()` | 8.4.1 | Motion vector prediction |

### reconstruct/
| Function | Spec Section | Description |
|----------|--------------|-------------|
| `reconstruct_mb()` | 8.5 | Transform and quantization |
| `clip_pixel()` | 8.5.14 | Picture construction |

### color/
| Function | Spec Section | Description |
|----------|--------------|-------------|
| `yuv_to_rgb()` | Annex E | Video usability information (VUI) |
| `upsample_chroma()` | - | 4:2:0 to 4:4:4 conversion |

## Key Tables from Spec

| Table | Section | Used In | Description |
|-------|---------|---------|-------------|
| Table 7-1 | 7.4.1 | bitstream/ | NAL unit types |
| Table 7-3 | 7.4.2.1 | parameters/ | Profile IDC values |
| Table 8-13 | 8.5.11 | dequant/ | LevelScale values |
| Table 9-5 | 9.2.1 | entropy/ | coeff_token VLC |
| Table 9-7 | 9.2.3 | entropy/ | total_zeros VLC |
| Table 9-10 | 9.2.4 | entropy/ | run_before VLC |

## Baseline Profile Constraints (Annex A.2.1)
- No B slices
- No CABAC (CAVLC only)
- No weighted prediction
- No 8x8 transform
- max_num_ref_frames ≤ number specified by level
