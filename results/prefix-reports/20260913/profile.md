# PyTorch microprofile

Separate from E2E measurements: the same 578-token prompt is requested three times, 16 output tokens each; GPU prefix cache is reset between requests. LRU stores once and loads twice; two-hit stores on the second request and loads on the third. GPU-only recomputes every time. Profiling changes execution overhead; these timings are diagnostic, not the throughput comparison.

Event duration sums can overlap and CPU operator categories can nest. Do not add categories into total latency or treat memcpy durations as end-to-end transfer time.

## gpu-cpu-l2: API

| Category | Events | Sum duration ms |
|---|---:|---:|
| Trace | 1 | 2055.920 |
| python_function | 27075 | 74271.037 |

## gpu-cpu-l2: GPU worker

| Category | Events | Sum duration ms |
|---|---:|---:|
| Trace | 1 | 2087.594 |
| cpu_op | 68182 | 2749.078 |
| cuda_driver | 563 | 9.888 |
| cuda_runtime | 17196 | 720.226 |
| gpu_memcpy | 1821 | 6.075 |
| gpu_memset | 132 | 0.121 |
| kernel | 18626 | 991.997 |
| overhead | 6 | 2.959 |
| python_function | 151684 | 113566.158 |

| CUDA memcpy event | Events | Sum duration ms | MiB |
|---|---:|---:|---:|
| Memcpy DtoD (Device -> Device) | 1260 | 1.242 | 3.6914 |
| Memcpy DtoH (Device -> Pageable) | 76 | 2.299 | 14.0004 |
| Memcpy HtoD (Pageable -> Device) | 485 | 2.534 | 28.3837 |

Top CUDA kernels:

| Kernel | Events | Sum duration ms |
|---|---:|---:|
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 6, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` | 2568 | 434.275 |
| `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn` | 140 | 271.046 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 7, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` | 1260 | 159.915 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8::Params)` | 1260 | 37.998 |
| `ampere_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_stages_32x1_tn` | 28 | 22.162 |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t> >, true, false, false, false, true, false, false, false>(flash::Flash_fwd_params)` | 84 | 14.230 |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t> >, false, false, false, false, true, false, true, false>(flash::Flash_fwd_params)` | 1260 | 10.935 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_f16_256x64_32x4_tn_align8>(cutlass_80_tensorop_s16816gemm_f16_256x64_32x4_tn_align8::Params)` | 56 | 7.019 |
| `triton_poi_fused_mul_silu_1` | 1344 | 5.858 |
| `void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t> >, 4, 3, true>(flash::Flash_fwd_params)` | 1260 | 4.621 |

## gpu-cpu-l2-two-hit: API

| Category | Events | Sum duration ms |
|---|---:|---:|
| Trace | 1 | 1945.286 |
| python_function | 27182 | 70286.047 |

## gpu-cpu-l2-two-hit: GPU worker

| Category | Events | Sum duration ms |
|---|---:|---:|
| Trace | 1 | 1973.194 |
| cpu_op | 67086 | 1782.173 |
| cuda_driver | 535 | 5.936 |
| cuda_runtime | 17139 | 631.834 |
| gpu_memcpy | 1765 | 3.756 |
| gpu_memset | 216 | 0.152 |
| kernel | 18570 | 932.447 |
| overhead | 6 | 2.649 |
| python_function | 149445 | 101404.420 |

| CUDA memcpy event | Events | Sum duration ms | MiB |
|---|---:|---:|---:|
| Memcpy DtoD (Device -> Device) | 1260 | 1.251 | 3.6914 |
| Memcpy DtoH (Device -> Pageable) | 76 | 1.176 | 14.0004 |
| Memcpy HtoD (Pageable -> Device) | 429 | 1.329 | 14.2841 |

Top CUDA kernels:

| Kernel | Events | Sum duration ms |
|---|---:|---:|
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 6, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` | 2568 | 415.496 |
| `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn` | 196 | 251.942 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 7, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` | 1260 | 148.359 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8::Params)` | 1260 | 36.597 |
| `ampere_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_stages_32x1_tn` | 56 | 20.580 |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t> >, true, false, false, false, true, false, false, false>(flash::Flash_fwd_params)` | 84 | 11.795 |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t> >, false, false, false, false, true, false, true, false>(flash::Flash_fwd_params)` | 1260 | 10.918 |
| `triton_poi_fused_mul_silu_1` | 1344 | 7.742 |
| `void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t> >, 4, 3, true>(flash::Flash_fwd_params)` | 1260 | 4.650 |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2` | 1344 | 3.678 |

## gpu-only: API

| Category | Events | Sum duration ms |
|---|---:|---:|
| Trace | 1 | 964.900 |
| python_function | 26880 | 34983.936 |

## gpu-only: GPU worker

| Category | Events | Sum duration ms |
|---|---:|---:|
| Trace | 1 | 984.245 |
| cpu_op | 65232 | 1305.751 |
| cuda_driver | 507 | 4.628 |
| cuda_runtime | 16939 | 456.105 |
| gpu_memcpy | 1653 | 1.373 |
| gpu_memset | 300 | 0.200 |
| kernel | 18486 | 777.071 |
| overhead | 4 | 4.389 |
| python_function | 118050 | 57352.072 |

| CUDA memcpy event | Events | Sum duration ms | MiB |
|---|---:|---:|---:|
| Memcpy DtoD (Device -> Device) | 1260 | 1.218 | 3.6914 |
| Memcpy DtoH (Device -> Pageable) | 48 | 0.040 | 0.0004 |
| Memcpy HtoD (Pageable -> Device) | 345 | 0.116 | 0.0751 |

Top CUDA kernels:

| Kernel | Events | Sum duration ms |
|---|---:|---:|
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 6, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` | 2568 | 415.295 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 7, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` | 1260 | 152.484 |
| `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn` | 252 | 118.903 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8::Params)` | 1260 | 35.449 |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t> >, false, false, false, false, true, false, true, false>(flash::Flash_fwd_params)` | 1260 | 10.463 |
| `ampere_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_stages_32x1_tn` | 84 | 9.774 |
| `triton_poi_fused_mul_silu_1` | 1344 | 9.321 |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t> >, true, false, false, false, true, false, false, false>(flash::Flash_fwd_params)` | 84 | 5.306 |
| `void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t> >, 4, 3, true>(flash::Flash_fwd_params)` | 1260 | 4.522 |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2` | 1344 | 2.968 |
