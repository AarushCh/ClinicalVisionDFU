| checkpoint | arch | input_size | params_m | file_mb | latency_ms_median | latency_ms_p95 | throughput_img_s |
|---|---|---|---|---|---|---|---|
| dfu_efficientnet_b0.pt | efficientnet_b0 | 224 | 4.01 | 16.34 | 9.93 | 14.44 | 100.7 |
| dfu_resnet18.pt | resnet18 | 224 | 11.18 | 44.79 | 9.38 | 10.95 | 106.6 |
| dfu_resnet18_grouped.pt | resnet18 | 224 | 11.18 | 44.79 | 9.82 | 11.64 | 101.8 |
| dfu_resnet18_grouped_dedup.pt | resnet18 | 224 | 11.18 | 44.79 | 9.36 | 11.84 | 106.9 |
| dfu_resnet18_grouped_gray.pt | resnet18 | 224 | 11.18 | 44.79 | 9.54 | 11.52 | 104.8 |
| dfu_resnet50.pt | resnet50 | 224 | 23.51 | 94.36 | 23.16 | 28.02 | 43.2 |
| dfu_model.pt | resnet50 | 384 | 23.51 | 94.37 | 47.33 | 55.37 | 21.1 |
