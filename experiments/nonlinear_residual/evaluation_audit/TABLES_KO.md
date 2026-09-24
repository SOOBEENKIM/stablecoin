# A5 수치 비교표

범위는 다섯 시간 정렬 시나리오의 최솟값~최댓값이며 신뢰구간이 아니다. UTC 두 시나리오는 core 자료가 같아 독립 반복이 아니다.

## 동시점 USDT 설명오차/잔차 RMS: Jan_Mar

| model | rmse_bp | reference_reconstruction_mse |
| --- | --- | --- |
| sequential_ols | 9.627 ~ 10.050 | — |
| pca_1 | 9.549 ~ 9.966 | 0.006 ~ 0.007 |
| pca_2 | 6.925 ~ 7.064 | 0.001 ~ 0.001 |
| ae_selected | 8.689 ~ 9.345 | 0.002 ~ 0.003 |

USDT 잔차 RMS 단위는 bp. 기준 5종 재구성 MSE는 표준화한 코인 프리미엄의 무차원 오차다. 서로 다른 지표다. Aug–Mar는 과거 검증 구간을 포함한 안정성 진단이다.

## 동시점 USDT 설명오차/잔차 RMS: Aug_Mar

| model | rmse_bp | reference_reconstruction_mse |
| --- | --- | --- |
| sequential_ols | 9.576 ~ 9.800 | — |
| pca_1 | 9.559 ~ 9.783 | 0.006 ~ 0.006 |
| pca_2 | 7.041 ~ 7.749 | 0.001 ~ 0.002 |
| ae_selected | 48.116 ~ 59.448 | 0.185 ~ 0.328 |

USDT 잔차 RMS 단위는 bp. 기준 5종 재구성 MSE는 표준화한 코인 프리미엄의 무차원 오차다. 서로 다른 지표다. Aug–Mar는 과거 검증 구간을 포함한 안정성 진단이다.

## 잔차 설명 probe: MSE 개선율(%)

| method | scope | algorithm | skill_vs_zero | skill_vs_past_mean |
| --- | --- | --- | --- | --- |
| ae_selected | all_references | lightgbm | -1.920 ~ 2.524 | 80.519 ~ 86.076 |
| ae_selected | all_references | ridge | -2308.971 ~ -1294.572 | -244.104 ~ -163.800 |
| ae_selected | mean_market | lightgbm | -4.100 ~ 1.767 | 80.023 ~ 85.968 |
| ae_selected | mean_market | ridge | -2225.175 ~ -1258.676 | -232.135 ~ -155.089 |
| pca_1 | all_references | lightgbm | 6.646 ~ 12.534 | 9.652 ~ 12.507 |
| pca_1 | all_references | ridge | 45.427 ~ 47.462 | 46.947 ~ 48.345 |
| pca_1 | mean_market | lightgbm | -4.798 ~ -3.247 | -3.704 ~ 0.571 |
| pca_1 | mean_market | ridge | -4.495 ~ -1.683 | -2.120 ~ 1.148 |
| pca_2 | all_references | lightgbm | -6.590 ~ -3.668 | -1.454 ~ 3.089 |
| pca_2 | all_references | ridge | -14.215 ~ -8.821 | -9.732 ~ -2.691 |
| pca_2 | mean_market | lightgbm | -5.469 ~ -2.164 | -1.200 ~ 4.510 |
| pca_2 | mean_market | ridge | -8.596 ~ -4.785 | -2.547 ~ 1.265 |
| sequential_ols | all_references | lightgbm | 7.206 ~ 12.524 | 9.641 ~ 12.263 |
| sequential_ols | all_references | ridge | 46.509 ~ 48.405 | 47.791 ~ 49.112 |
| sequential_ols | mean_market | lightgbm | -4.681 ~ -3.272 | -4.008 ~ 0.159 |
| sequential_ols | mean_market | ridge | -4.345 ~ -1.559 | -2.278 ~ 0.874 |

양수는 해당 기준보다 오차 감소, 음수는 악화다. 과거 평균 이동 때문에 두 기준이 달라지며 AE의 큰 과거 평균 대비 수치를 공통 요인 잔여 설명력으로 해석하지 않는다.

## observed_change / q=0.1

| method | info | algorithm | skill | below_rate |
| --- | --- | --- | --- | --- |
| baseline | without_positioning | linear_qr | -14.003 ~ -10.618 | 15.799 ~ 16.638 |
| baseline | without_positioning | quantile_lightgbm | -13.011 ~ -4.648 | 13.368 ~ 17.851 |
| baseline | with_positioning | linear_qr | -14.003 ~ -9.772 | 13.345 ~ 16.638 |
| baseline | with_positioning | quantile_lightgbm | -7.138 ~ -2.912 | 13.368 ~ 16.291 |
| sequential_ols | without_positioning | linear_qr | -15.131 ~ -10.373 | 16.118 ~ 16.984 |
| sequential_ols | without_positioning | quantile_lightgbm | -13.084 ~ -4.454 | 13.194 ~ 17.678 |
| sequential_ols | with_positioning | linear_qr | -15.131 ~ -9.489 | 13.345 ~ 16.984 |
| sequential_ols | with_positioning | quantile_lightgbm | -7.992 ~ -2.676 | 12.847 ~ 17.158 |
| pca_1 | without_positioning | linear_qr | -15.131 ~ -10.397 | 16.118 ~ 16.984 |
| pca_1 | without_positioning | quantile_lightgbm | -12.313 ~ -4.517 | 12.153 ~ 17.678 |
| pca_1 | with_positioning | linear_qr | -15.131 ~ -9.481 | 13.345 ~ 16.984 |
| pca_1 | with_positioning | quantile_lightgbm | -8.078 ~ -2.796 | 13.021 ~ 17.158 |
| pca_2 | without_positioning | linear_qr | -13.493 ~ -10.391 | 15.625 ~ 16.638 |
| pca_2 | without_positioning | quantile_lightgbm | -13.310 ~ -5.590 | 13.194 ~ 17.504 |
| pca_2 | with_positioning | linear_qr | -13.493 ~ -9.884 | 13.865 ~ 16.638 |
| pca_2 | with_positioning | quantile_lightgbm | -9.318 ~ -4.461 | 14.236 ~ 17.158 |
| ae_selected | without_positioning | linear_qr | -11.923 ~ -6.695 | 15.278 ~ 16.464 |
| ae_selected | without_positioning | quantile_lightgbm | -12.105 ~ -3.924 | 12.153 ~ 17.158 |
| ae_selected | with_positioning | linear_qr | -12.804 ~ -6.682 | 13.345 ~ 16.464 |
| ae_selected | with_positioning | quantile_lightgbm | -7.435 ~ -2.966 | 13.542 ~ 16.984 |

skill(%)는 해당 과거 분위수 기준 대비 pinball loss 감소율. below_rate(%)는 실제값이 예측 분위수보다 낮은 비율이다. own_residual은 목표가 달라 raw loss로 분해 모형의 순위를 매기지 않는다.

## observed_change / q=0.5

| method | info | algorithm | skill | below_rate |
| --- | --- | --- | --- | --- |
| baseline | without_positioning | linear_qr | -4.476 ~ 2.262 | 54.419 ~ 61.806 |
| baseline | without_positioning | quantile_lightgbm | -7.145 ~ 1.293 | 53.726 ~ 60.590 |
| baseline | with_positioning | linear_qr | -5.169 ~ 1.853 | 54.419 ~ 61.632 |
| baseline | with_positioning | quantile_lightgbm | -6.141 ~ 3.025 | 53.206 ~ 60.243 |
| sequential_ols | without_positioning | linear_qr | -3.951 ~ 2.262 | 54.593 ~ 61.285 |
| sequential_ols | without_positioning | quantile_lightgbm | -6.926 ~ 1.582 | 53.553 ~ 59.896 |
| sequential_ols | with_positioning | linear_qr | -4.092 ~ 1.853 | 54.593 ~ 60.243 |
| sequential_ols | with_positioning | quantile_lightgbm | -6.905 ~ 2.458 | 53.899 ~ 59.896 |
| pca_1 | without_positioning | linear_qr | -3.932 ~ 2.262 | 54.593 ~ 61.285 |
| pca_1 | without_positioning | quantile_lightgbm | -7.403 ~ 1.530 | 53.206 ~ 59.896 |
| pca_1 | with_positioning | linear_qr | -4.139 ~ 1.853 | 54.593 ~ 60.069 |
| pca_1 | with_positioning | quantile_lightgbm | -6.571 ~ 2.977 | 53.899 ~ 59.896 |
| pca_2 | without_positioning | linear_qr | -2.836 ~ 2.382 | 54.593 ~ 61.285 |
| pca_2 | without_positioning | quantile_lightgbm | -6.551 ~ 1.775 | 54.419 ~ 59.722 |
| pca_2 | with_positioning | linear_qr | -4.072 ~ 1.864 | 54.593 ~ 61.111 |
| pca_2 | with_positioning | quantile_lightgbm | -5.010 ~ 3.563 | 54.246 ~ 58.854 |
| ae_selected | without_positioning | linear_qr | -4.663 ~ 2.262 | 54.419 ~ 61.806 |
| ae_selected | without_positioning | quantile_lightgbm | -7.080 ~ 2.143 | 53.380 ~ 60.069 |
| ae_selected | with_positioning | linear_qr | -5.079 ~ 1.853 | 54.419 ~ 61.632 |
| ae_selected | with_positioning | quantile_lightgbm | -6.720 ~ 2.865 | 54.419 ~ 60.069 |

skill(%)는 해당 과거 분위수 기준 대비 pinball loss 감소율. below_rate(%)는 실제값이 예측 분위수보다 낮은 비율이다. own_residual은 목표가 달라 raw loss로 분해 모형의 순위를 매기지 않는다.

## own_residual / q=0.1

| method | info | algorithm | skill | below_rate |
| --- | --- | --- | --- | --- |
| sequential_ols | without_positioning | linear_qr | 0.789 ~ 5.276 | 15.104 ~ 16.464 |
| sequential_ols | without_positioning | quantile_lightgbm | -8.027 ~ -1.277 | 17.504 ~ 19.097 |
| sequential_ols | with_positioning | linear_qr | 0.825 ~ 5.276 | 15.104 ~ 16.464 |
| sequential_ols | with_positioning | quantile_lightgbm | -9.888 ~ -1.587 | 17.851 ~ 20.139 |
| pca_1 | without_positioning | linear_qr | 0.893 ~ 5.645 | 14.757 ~ 16.638 |
| pca_1 | without_positioning | quantile_lightgbm | -9.504 ~ -0.110 | 17.504 ~ 18.544 |
| pca_1 | with_positioning | linear_qr | 0.893 ~ 5.645 | 14.757 ~ 16.464 |
| pca_1 | with_positioning | quantile_lightgbm | -10.320 ~ -1.187 | 18.544 ~ 19.965 |
| pca_2 | without_positioning | linear_qr | 1.354 ~ 4.734 | 14.211 ~ 15.945 |
| pca_2 | without_positioning | quantile_lightgbm | -6.633 ~ -4.393 | 16.291 ~ 19.064 |
| pca_2 | with_positioning | linear_qr | 1.223 ~ 4.734 | 14.211 ~ 15.945 |
| pca_2 | with_positioning | quantile_lightgbm | -6.222 ~ 0.036 | 16.464 ~ 18.229 |
| ae_selected | without_positioning | linear_qr | -9.240 ~ 8.942 | 12.153 ~ 14.385 |
| ae_selected | without_positioning | quantile_lightgbm | -14.150 ~ -1.539 | 15.945 ~ 18.024 |
| ae_selected | with_positioning | linear_qr | -9.240 ~ 8.906 | 12.153 ~ 14.385 |
| ae_selected | with_positioning | quantile_lightgbm | -13.413 ~ -0.753 | 15.078 ~ 18.229 |

skill(%)는 해당 과거 분위수 기준 대비 pinball loss 감소율. below_rate(%)는 실제값이 예측 분위수보다 낮은 비율이다. own_residual은 목표가 달라 raw loss로 분해 모형의 순위를 매기지 않는다.

## own_residual / q=0.5

| method | info | algorithm | skill | below_rate |
| --- | --- | --- | --- | --- |
| sequential_ols | without_positioning | linear_qr | 0.242 ~ 6.854 | 50.780 ~ 58.333 |
| sequential_ols | without_positioning | quantile_lightgbm | 1.072 ~ 5.045 | 53.206 ~ 59.722 |
| sequential_ols | with_positioning | linear_qr | 0.694 ~ 6.844 | 50.953 ~ 58.333 |
| sequential_ols | with_positioning | quantile_lightgbm | 1.257 ~ 4.927 | 53.899 ~ 59.549 |
| pca_1 | without_positioning | linear_qr | 0.253 ~ 7.327 | 51.127 ~ 58.333 |
| pca_1 | without_positioning | quantile_lightgbm | 1.467 ~ 5.391 | 53.553 ~ 59.201 |
| pca_1 | with_positioning | linear_qr | 0.758 ~ 7.333 | 51.300 ~ 58.333 |
| pca_1 | with_positioning | quantile_lightgbm | 1.351 ~ 5.309 | 53.206 ~ 60.590 |
| pca_2 | without_positioning | linear_qr | 0.212 ~ 8.264 | 50.953 ~ 55.035 |
| pca_2 | without_positioning | quantile_lightgbm | -0.192 ~ 5.790 | 52.340 ~ 59.792 |
| pca_2 | with_positioning | linear_qr | 0.212 ~ 8.306 | 50.953 ~ 55.556 |
| pca_2 | with_positioning | quantile_lightgbm | 0.144 ~ 6.130 | 52.860 ~ 59.549 |
| ae_selected | without_positioning | linear_qr | -3.671 ~ 7.146 | 46.620 ~ 54.593 |
| ae_selected | without_positioning | quantile_lightgbm | 4.508 ~ 9.097 | 56.672 ~ 62.674 |
| ae_selected | with_positioning | linear_qr | -3.659 ~ 7.150 | 46.274 ~ 54.419 |
| ae_selected | with_positioning | quantile_lightgbm | 5.254 ~ 9.700 | 57.539 ~ 63.542 |

skill(%)는 해당 과거 분위수 기준 대비 pinball loss 감소율. below_rate(%)는 실제값이 예측 분위수보다 낮은 비율이다. own_residual은 목표가 달라 raw loss로 분해 모형의 순위를 매기지 않는다.

## 동일 관측 목표 q10의 paired 손실 차이

| scenario | method | info | algorithm | contrast | relative_pct | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clock_UTC_quote | sequential_ols | without_positioning | linear_qr | add_residual | 0.00000 | 0.00000 | 0.00000 |
| clock_UTC_quote | sequential_ols | without_positioning | quantile_lightgbm | add_residual | 0.06401 | -0.18269 | 0.20119 |
| clock_UTC_quote | sequential_ols | with_positioning | linear_qr | add_residual | 0.00000 | 0.00000 | 0.00000 |
| clock_UTC_quote | sequential_ols | with_positioning | quantile_lightgbm | add_residual | 0.79672 | 0.00311 | 0.14208 |
| clock_UTC_quote | pca_1 | without_positioning | linear_qr | add_residual | 0.00000 | 0.00000 | 0.00000 |
| clock_UTC_quote | pca_1 | without_positioning | quantile_lightgbm | add_residual | -0.61817 | -0.21761 | 0.09852 |
| clock_UTC_quote | pca_1 | with_positioning | linear_qr | add_residual | 0.00000 | 0.00000 | 0.00000 |
| clock_UTC_quote | pca_1 | with_positioning | quantile_lightgbm | add_residual | 0.87724 | 0.00222 | 0.16248 |
| clock_UTC_quote | pca_2 | without_positioning | linear_qr | add_residual | 0.37902 | -0.00616 | 0.09624 |
| clock_UTC_quote | pca_2 | without_positioning | quantile_lightgbm | add_residual | 0.26445 | -0.19481 | 0.22906 |
| clock_UTC_quote | pca_2 | with_positioning | linear_qr | add_residual | 0.12896 | -0.04536 | 0.08208 |
| clock_UTC_quote | pca_2 | with_positioning | quantile_lightgbm | add_residual | 2.03429 | 0.09185 | 0.27226 |
| clock_UTC_quote | ae_selected | without_positioning | linear_qr | add_residual | 0.00000 | 0.00000 | 0.00000 |
| clock_UTC_quote | ae_selected | without_positioning | linear_qr | AE_minus_pca_2 | -0.37759 | -0.09624 | 0.00616 |
| clock_UTC_quote | ae_selected | without_positioning | quantile_lightgbm | add_residual | -0.80182 | -0.36990 | 0.20344 |
| clock_UTC_quote | ae_selected | without_positioning | quantile_lightgbm | AE_minus_pca_2 | -1.06345 | -0.31972 | 0.12078 |
| clock_UTC_quote | ae_selected | with_positioning | linear_qr | add_residual | 0.00000 | 0.00000 | 0.00000 |
| clock_UTC_quote | ae_selected | with_positioning | linear_qr | AE_minus_pca_2 | -0.12879 | -0.08208 | 0.04536 |
| clock_UTC_quote | ae_selected | with_positioning | quantile_lightgbm | add_residual | 0.27670 | -0.04357 | 0.09764 |
| clock_UTC_quote | ae_selected | with_positioning | quantile_lightgbm | AE_minus_pca_2 | -1.72255 | -0.26266 | -0.05722 |
| clock_NY_quote | sequential_ols | without_positioning | linear_qr | add_residual | 0.98939 | -0.00543 | 0.15036 |
| clock_NY_quote | sequential_ols | without_positioning | quantile_lightgbm | add_residual | 2.43185 | 0.05496 | 0.26840 |
| clock_NY_quote | sequential_ols | with_positioning | linear_qr | add_residual | 0.98939 | -0.00543 | 0.15036 |
| clock_NY_quote | sequential_ols | with_positioning | quantile_lightgbm | add_residual | 0.14557 | -0.08778 | 0.09694 |
| clock_NY_quote | pca_1 | without_positioning | linear_qr | add_residual | 0.99009 | -0.00572 | 0.15099 |
| clock_NY_quote | pca_1 | without_positioning | quantile_lightgbm | add_residual | 2.99899 | 0.05994 | 0.32347 |
| clock_NY_quote | pca_1 | with_positioning | linear_qr | add_residual | 0.99009 | -0.00572 | 0.15099 |
| clock_NY_quote | pca_1 | with_positioning | quantile_lightgbm | add_residual | 0.99968 | -0.04203 | 0.16826 |
| clock_NY_quote | pca_2 | without_positioning | linear_qr | add_residual | -0.44723 | -0.10466 | 0.05437 |
| clock_NY_quote | pca_2 | without_positioning | quantile_lightgbm | add_residual | 2.08062 | 0.03528 | 0.22850 |
| clock_NY_quote | pca_2 | with_positioning | linear_qr | add_residual | -0.44723 | -0.10466 | 0.05437 |
| clock_NY_quote | pca_2 | with_positioning | quantile_lightgbm | add_residual | 1.83461 | 0.02607 | 0.23153 |
| clock_NY_quote | ae_selected | without_positioning | linear_qr | add_residual | -1.82464 | -0.17187 | -0.07354 |
| clock_NY_quote | ae_selected | without_positioning | linear_qr | AE_minus_pca_2 | -1.38360 | -0.16981 | -0.02830 |
| clock_NY_quote | ae_selected | without_positioning | quantile_lightgbm | add_residual | 1.45456 | 0.02126 | 0.16719 |
| clock_NY_quote | ae_selected | without_positioning | quantile_lightgbm | AE_minus_pca_2 | -0.61331 | -0.12895 | 0.05299 |
| clock_NY_quote | ae_selected | with_positioning | linear_qr | add_residual | -1.82464 | -0.17187 | -0.07354 |
| clock_NY_quote | ae_selected | with_positioning | linear_qr | AE_minus_pca_2 | -1.38360 | -0.16981 | -0.02830 |
| clock_NY_quote | ae_selected | with_positioning | quantile_lightgbm | add_residual | 0.86363 | -0.01323 | 0.13458 |
| clock_NY_quote | ae_selected | with_positioning | quantile_lightgbm | AE_minus_pca_2 | -0.95349 | -0.19587 | 0.05795 |
| clock_FXUTC_USNY_quote | sequential_ols | without_positioning | linear_qr | add_residual | 0.00000 | 0.00000 | 0.00000 |
| clock_FXUTC_USNY_quote | sequential_ols | without_positioning | quantile_lightgbm | add_residual | 0.06401 | -0.18269 | 0.20119 |
| clock_FXUTC_USNY_quote | sequential_ols | with_positioning | linear_qr | add_residual | 0.00000 | 0.00000 | 0.00000 |
| clock_FXUTC_USNY_quote | sequential_ols | with_positioning | quantile_lightgbm | add_residual | 0.79672 | 0.00311 | 0.14208 |
| clock_FXUTC_USNY_quote | pca_1 | without_positioning | linear_qr | add_residual | 0.00000 | 0.00000 | 0.00000 |
| clock_FXUTC_USNY_quote | pca_1 | without_positioning | quantile_lightgbm | add_residual | -0.61817 | -0.21761 | 0.09852 |
| clock_FXUTC_USNY_quote | pca_1 | with_positioning | linear_qr | add_residual | 0.00000 | 0.00000 | 0.00000 |
| clock_FXUTC_USNY_quote | pca_1 | with_positioning | quantile_lightgbm | add_residual | 0.87724 | 0.00222 | 0.16248 |
| clock_FXUTC_USNY_quote | pca_2 | without_positioning | linear_qr | add_residual | 0.37902 | -0.00616 | 0.09624 |
| clock_FXUTC_USNY_quote | pca_2 | without_positioning | quantile_lightgbm | add_residual | 0.26445 | -0.19481 | 0.22906 |
| clock_FXUTC_USNY_quote | pca_2 | with_positioning | linear_qr | add_residual | 0.12896 | -0.04536 | 0.08208 |
| clock_FXUTC_USNY_quote | pca_2 | with_positioning | quantile_lightgbm | add_residual | 2.03429 | 0.09185 | 0.27226 |
| clock_FXUTC_USNY_quote | ae_selected | without_positioning | linear_qr | add_residual | 0.00000 | 0.00000 | 0.00000 |
| clock_FXUTC_USNY_quote | ae_selected | without_positioning | linear_qr | AE_minus_pca_2 | -0.37759 | -0.09624 | 0.00616 |
| clock_FXUTC_USNY_quote | ae_selected | without_positioning | quantile_lightgbm | add_residual | -0.80182 | -0.36990 | 0.20344 |
| clock_FXUTC_USNY_quote | ae_selected | without_positioning | quantile_lightgbm | AE_minus_pca_2 | -1.06345 | -0.31972 | 0.12078 |
| clock_FXUTC_USNY_quote | ae_selected | with_positioning | linear_qr | add_residual | 0.00000 | 0.00000 | 0.00000 |
| clock_FXUTC_USNY_quote | ae_selected | with_positioning | linear_qr | AE_minus_pca_2 | -0.12879 | -0.08208 | 0.04536 |
| clock_FXUTC_USNY_quote | ae_selected | with_positioning | quantile_lightgbm | add_residual | 0.27670 | -0.04357 | 0.09764 |
| clock_FXUTC_USNY_quote | ae_selected | with_positioning | quantile_lightgbm | AE_minus_pca_2 | -1.72255 | -0.26266 | -0.05722 |
| clock_FXKST_USNY_quote | sequential_ols | without_positioning | linear_qr | add_residual | -0.22078 | -0.13240 | 0.09084 |
| clock_FXKST_USNY_quote | sequential_ols | without_positioning | quantile_lightgbm | add_residual | 1.54372 | -0.04144 | 0.33312 |
| clock_FXKST_USNY_quote | sequential_ols | with_positioning | linear_qr | add_residual | -0.25750 | -0.13300 | 0.08187 |
| clock_FXKST_USNY_quote | sequential_ols | with_positioning | quantile_lightgbm | add_residual | 1.43975 | -0.02056 | 0.30272 |
| clock_FXKST_USNY_quote | pca_1 | without_positioning | linear_qr | add_residual | -0.19965 | -0.12974 | 0.09200 |
| clock_FXKST_USNY_quote | pca_1 | without_positioning | quantile_lightgbm | add_residual | 1.49324 | -0.03013 | 0.31925 |
| clock_FXKST_USNY_quote | pca_1 | with_positioning | linear_qr | add_residual | -0.26520 | -0.13476 | 0.08199 |
| clock_FXKST_USNY_quote | pca_1 | with_positioning | quantile_lightgbm | add_residual | 3.32245 | 0.04248 | 0.57262 |
| clock_FXKST_USNY_quote | pca_2 | without_positioning | linear_qr | add_residual | 0.22108 | -0.04433 | 0.09046 |
| clock_FXKST_USNY_quote | pca_2 | without_positioning | quantile_lightgbm | add_residual | 1.31460 | 0.01184 | 0.22801 |
| clock_FXKST_USNY_quote | pca_2 | with_positioning | linear_qr | add_residual | 0.10175 | -0.05516 | 0.07737 |
| clock_FXKST_USNY_quote | pca_2 | with_positioning | quantile_lightgbm | add_residual | 2.28012 | 0.05108 | 0.38406 |
| clock_FXKST_USNY_quote | ae_selected | without_positioning | linear_qr | add_residual | -3.54632 | -0.56681 | -0.11521 |
| clock_FXKST_USNY_quote | ae_selected | without_positioning | linear_qr | AE_minus_pca_2 | -3.75909 | -0.61016 | -0.12564 |
| clock_FXKST_USNY_quote | ae_selected | without_positioning | quantile_lightgbm | add_residual | -0.76775 | -0.18527 | 0.03960 |
| clock_FXKST_USNY_quote | ae_selected | without_positioning | quantile_lightgbm | AE_minus_pca_2 | -2.05533 | -0.32035 | -0.06570 |
| clock_FXKST_USNY_quote | ae_selected | with_positioning | linear_qr | add_residual | -2.81490 | -0.49068 | -0.04737 |
| clock_FXKST_USNY_quote | ae_selected | with_positioning | linear_qr | AE_minus_pca_2 | -2.91369 | -0.50245 | -0.05843 |
| clock_FXKST_USNY_quote | ae_selected | with_positioning | quantile_lightgbm | add_residual | 1.35329 | -0.05188 | 0.31662 |
| clock_FXKST_USNY_quote | ae_selected | with_positioning | quantile_lightgbm | AE_minus_pca_2 | -0.90616 | -0.22492 | 0.06889 |
| clock_NY_macro_end1h_quote | sequential_ols | without_positioning | linear_qr | add_residual | -0.27355 | -0.13420 | 0.09836 |
| clock_NY_macro_end1h_quote | sequential_ols | without_positioning | quantile_lightgbm | add_residual | -0.18606 | -0.08545 | 0.06373 |
| clock_NY_macro_end1h_quote | sequential_ols | with_positioning | linear_qr | add_residual | 1.32820 | -0.05883 | 0.21409 |
| clock_NY_macro_end1h_quote | sequential_ols | with_positioning | quantile_lightgbm | add_residual | -0.22866 | -0.07714 | 0.05487 |
| clock_NY_macro_end1h_quote | pca_1 | without_positioning | linear_qr | add_residual | -0.30898 | -0.13850 | 0.09715 |
| clock_NY_macro_end1h_quote | pca_1 | without_positioning | quantile_lightgbm | add_residual | -0.12515 | -0.09715 | 0.07523 |
| clock_NY_macro_end1h_quote | pca_1 | with_positioning | linear_qr | add_residual | 1.29220 | -0.06206 | 0.21232 |
| clock_NY_macro_end1h_quote | pca_1 | with_positioning | quantile_lightgbm | add_residual | -0.11265 | -0.07635 | 0.06632 |
| clock_NY_macro_end1h_quote | pca_2 | without_positioning | linear_qr | add_residual | -2.53948 | -0.36877 | 0.01992 |
| clock_NY_macro_end1h_quote | pca_2 | without_positioning | quantile_lightgbm | add_residual | 0.90010 | -0.10265 | 0.21157 |
| clock_NY_macro_end1h_quote | pca_2 | with_positioning | linear_qr | add_residual | -0.98701 | -0.26731 | 0.10734 |
| clock_NY_macro_end1h_quote | pca_2 | with_positioning | quantile_lightgbm | add_residual | 1.50590 | -0.04151 | 0.21521 |
| clock_NY_macro_end1h_quote | ae_selected | without_positioning | linear_qr | add_residual | -2.16203 | -0.19237 | -0.06049 |
| clock_NY_macro_end1h_quote | ae_selected | without_positioning | linear_qr | AE_minus_pca_2 | 0.38729 | -0.14120 | 0.22635 |
| clock_NY_macro_end1h_quote | ae_selected | without_positioning | quantile_lightgbm | add_residual | -0.69186 | -0.09500 | 0.01779 |
| clock_NY_macro_end1h_quote | ae_selected | without_positioning | quantile_lightgbm | AE_minus_pca_2 | -1.57775 | -0.20391 | 0.01981 |
| clock_NY_macro_end1h_quote | ae_selected | with_positioning | linear_qr | add_residual | -0.23675 | -0.02062 | -0.00636 |
| clock_NY_macro_end1h_quote | ae_selected | with_positioning | linear_qr | AE_minus_pca_2 | 0.75774 | -0.12259 | 0.25161 |
| clock_NY_macro_end1h_quote | ae_selected | with_positioning | quantile_lightgbm | add_residual | 0.05267 | -0.05016 | 0.05997 |
| clock_NY_macro_end1h_quote | ae_selected | with_positioning | quantile_lightgbm | AE_minus_pca_2 | -1.43168 | -0.19256 | 0.02234 |

relative_pct는 손실 증감률, CI는 bp 단위 pinball loss 차이다. 음수는 앞의 방법이 개선. 고정 예측 2,000회 7일 블록 구간이며 재학습·선택 불확실성 전체를 포함하지 않는다.

## 원래 연구 M2: Jan–Mar, h=6, UTC 기준

| method | q | term | n | effect | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| sequential_ols | 0.100 | e_origin | 577 | 0.654 | -0.464 | 2.122 |
| sequential_ols | 0.100 | downside | 577 | -1.424 | -4.071 | 2.624 |
| sequential_ols | 0.100 | account_ls | 577 | 1.127 | -0.211 | 3.287 |
| sequential_ols | 0.500 | e_origin | 577 | 0.751 | -0.031 | 1.959 |
| sequential_ols | 0.500 | downside | 577 | -2.900 | -5.487 | -0.626 |
| sequential_ols | 0.500 | account_ls | 577 | 2.014 | 0.843 | 3.256 |
| pca_1 | 0.100 | e_origin | 577 | 0.725 | -0.476 | 2.117 |
| pca_1 | 0.100 | downside | 577 | -1.439 | -4.064 | 2.848 |
| pca_1 | 0.100 | account_ls | 577 | 1.273 | -0.211 | 3.251 |
| pca_1 | 0.500 | e_origin | 577 | 0.766 | 0.013 | 1.965 |
| pca_1 | 0.500 | downside | 577 | -2.900 | -5.644 | -0.526 |
| pca_1 | 0.500 | account_ls | 577 | 2.066 | 0.790 | 3.260 |
| pca_2 | 0.100 | e_origin | 577 | 0.751 | 0.006 | 1.508 |
| pca_2 | 0.100 | downside | 577 | -0.987 | -2.889 | 0.653 |
| pca_2 | 0.100 | account_ls | 577 | 1.042 | 0.067 | 2.412 |
| pca_2 | 0.500 | e_origin | 577 | 1.279 | 0.309 | 2.110 |
| pca_2 | 0.500 | downside | 577 | -1.986 | -3.541 | -0.497 |
| pca_2 | 0.500 | account_ls | 577 | 1.611 | 0.732 | 2.701 |
| ae_selected | 0.100 | e_origin | 577 | 0.592 | -0.876 | 2.338 |
| ae_selected | 0.100 | downside | 577 | -1.879 | -4.772 | 0.554 |
| ae_selected | 0.100 | account_ls | 577 | -0.729 | -2.653 | 1.802 |
| ae_selected | 0.500 | e_origin | 577 | 0.933 | 0.256 | 2.394 |
| ae_selected | 0.500 | downside | 577 | -1.944 | -4.540 | -0.252 |
| ae_selected | 0.500 | account_ls | 577 | 0.533 | -1.411 | 1.767 |

효과 단위: 원래 평가표본에서 설명변수 1표준편차당 bp. 구간: 7일 달력 블록, 199회, PCA/AE 인코더부터 재학습. AE 월별 구조/규제 선택은 고정했다.

## 시각 가정별 q10 계수

| scenario | method | term | effect | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| clock_UTC_quote | sequential_ols | downside | -1.424 | -4.071 | 2.624 |
| clock_UTC_quote | sequential_ols | account_ls | 1.127 | -0.211 | 3.287 |
| clock_UTC_quote | pca_1 | downside | -1.439 | -4.064 | 2.848 |
| clock_UTC_quote | pca_1 | account_ls | 1.273 | -0.211 | 3.251 |
| clock_UTC_quote | pca_2 | downside | -0.987 | -2.889 | 0.653 |
| clock_UTC_quote | pca_2 | account_ls | 1.042 | 0.067 | 2.412 |
| clock_UTC_quote | ae_selected | downside | -1.879 | -4.772 | 0.554 |
| clock_UTC_quote | ae_selected | account_ls | -0.729 | -2.653 | 1.802 |
| clock_NY_quote | sequential_ols | downside | 1.170 | -2.510 | 4.286 |
| clock_NY_quote | sequential_ols | account_ls | 1.617 | -0.029 | 3.251 |
| clock_NY_quote | pca_1 | downside | 1.191 | -2.255 | 4.224 |
| clock_NY_quote | pca_1 | account_ls | 1.578 | -0.022 | 3.228 |
| clock_NY_quote | pca_2 | downside | 0.594 | -2.681 | 2.728 |
| clock_NY_quote | pca_2 | account_ls | 1.014 | 0.097 | 2.452 |
| clock_NY_quote | ae_selected | downside | 0.754 | -2.228 | 2.712 |
| clock_NY_quote | ae_selected | account_ls | -0.459 | -3.175 | 1.619 |
| clock_FXKST_USNY_quote | sequential_ols | downside | -1.585 | -4.252 | 1.601 |
| clock_FXKST_USNY_quote | sequential_ols | account_ls | 0.226 | -1.113 | 1.578 |
| clock_FXKST_USNY_quote | pca_1 | downside | -1.744 | -4.347 | 1.673 |
| clock_FXKST_USNY_quote | pca_1 | account_ls | 0.328 | -1.172 | 1.607 |
| clock_FXKST_USNY_quote | pca_2 | downside | -0.203 | -1.577 | 2.748 |
| clock_FXKST_USNY_quote | pca_2 | account_ls | -0.121 | -2.053 | 1.224 |
| clock_FXKST_USNY_quote | ae_selected | downside | -0.975 | -3.566 | 1.108 |
| clock_FXKST_USNY_quote | ae_selected | account_ls | -1.590 | -2.793 | 0.558 |
| clock_NY_macro_end1h_quote | sequential_ols | downside | 0.904 | -3.543 | 4.285 |
| clock_NY_macro_end1h_quote | sequential_ols | account_ls | 1.553 | -0.518 | 3.128 |
| clock_NY_macro_end1h_quote | pca_1 | downside | 0.889 | -3.448 | 4.173 |
| clock_NY_macro_end1h_quote | pca_1 | account_ls | 1.545 | -0.511 | 3.076 |
| clock_NY_macro_end1h_quote | pca_2 | downside | 0.030 | -4.040 | 3.140 |
| clock_NY_macro_end1h_quote | pca_2 | account_ls | 1.111 | -0.206 | 2.528 |
| clock_NY_macro_end1h_quote | ae_selected | downside | 1.064 | -3.503 | 2.614 |
| clock_NY_macro_end1h_quote | ae_selected | account_ls | -1.520 | -3.362 | 1.727 |

## 하방 분위수 특이성·시차별 차이

| scenario | method | term | contrast | effect | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| clock_UTC_quote | ae_selected | account_ls | q10_minus_q50_h6 | -1.263 | -2.247 | 1.026 |
| clock_UTC_quote | ae_selected | account_ls | h6_minus_h1_q10 | -0.299 | -4.348 | 1.902 |
| clock_UTC_quote | ae_selected | account_ls | h12_minus_h1_q10 | -0.461 | -4.745 | 1.067 |
| clock_UTC_quote | ae_selected | downside | q10_minus_q50_h6 | 0.065 | -1.772 | 2.706 |
| clock_UTC_quote | ae_selected | downside | h6_minus_h1_q10 | -1.720 | -4.821 | 7.062 |
| clock_UTC_quote | ae_selected | downside | h12_minus_h1_q10 | -1.554 | -3.308 | 9.363 |
| clock_UTC_quote | pca_1 | account_ls | q10_minus_q50_h6 | -0.793 | -2.417 | 0.693 |
| clock_UTC_quote | pca_1 | account_ls | h6_minus_h1_q10 | -1.592 | -4.066 | 2.091 |
| clock_UTC_quote | pca_1 | account_ls | h12_minus_h1_q10 | -2.132 | -4.590 | 1.506 |
| clock_UTC_quote | pca_1 | downside | q10_minus_q50_h6 | 1.461 | -1.409 | 5.421 |
| clock_UTC_quote | pca_1 | downside | h6_minus_h1_q10 | -0.242 | -6.328 | 7.631 |
| clock_UTC_quote | pca_1 | downside | h12_minus_h1_q10 | 0.447 | -2.297 | 10.050 |
| clock_UTC_quote | pca_2 | account_ls | q10_minus_q50_h6 | -0.569 | -1.402 | 0.703 |
| clock_UTC_quote | pca_2 | account_ls | h6_minus_h1_q10 | -0.673 | -3.139 | 2.640 |
| clock_UTC_quote | pca_2 | account_ls | h12_minus_h1_q10 | -1.766 | -4.317 | 3.128 |
| clock_UTC_quote | pca_2 | downside | q10_minus_q50_h6 | 0.999 | -0.939 | 2.490 |
| clock_UTC_quote | pca_2 | downside | h6_minus_h1_q10 | 0.846 | -3.112 | 7.200 |
| clock_UTC_quote | pca_2 | downside | h12_minus_h1_q10 | 2.748 | -2.410 | 10.032 |
| clock_UTC_quote | sequential_ols | account_ls | q10_minus_q50_h6 | -0.888 | -2.518 | 0.727 |
| clock_UTC_quote | sequential_ols | account_ls | h6_minus_h1_q10 | -1.679 | -4.133 | 2.127 |
| clock_UTC_quote | sequential_ols | account_ls | h12_minus_h1_q10 | -2.161 | -4.484 | 1.608 |
| clock_UTC_quote | sequential_ols | downside | q10_minus_q50_h6 | 1.476 | -1.216 | 5.598 |
| clock_UTC_quote | sequential_ols | downside | h6_minus_h1_q10 | -0.075 | -6.319 | 7.921 |
| clock_UTC_quote | sequential_ols | downside | h12_minus_h1_q10 | 0.629 | -2.344 | 10.205 |
| clock_NY_quote | ae_selected | account_ls | q10_minus_q50_h6 | 0.657 | -2.025 | 1.442 |
| clock_NY_quote | ae_selected | account_ls | h6_minus_h1_q10 | -0.198 | -2.982 | 2.953 |
| clock_NY_quote | ae_selected | account_ls | h12_minus_h1_q10 | -1.194 | -2.741 | 2.077 |
| clock_NY_quote | ae_selected | downside | q10_minus_q50_h6 | 1.275 | -1.331 | 3.752 |
| clock_NY_quote | ae_selected | downside | h6_minus_h1_q10 | 0.672 | -3.019 | 9.945 |
| clock_NY_quote | ae_selected | downside | h12_minus_h1_q10 | 4.724 | -0.572 | 12.036 |
| clock_NY_quote | pca_1 | account_ls | q10_minus_q50_h6 | 0.752 | -1.004 | 2.174 |
| clock_NY_quote | pca_1 | account_ls | h6_minus_h1_q10 | -1.941 | -5.410 | 1.679 |
| clock_NY_quote | pca_1 | account_ls | h12_minus_h1_q10 | 1.056 | -3.025 | 2.604 |
| clock_NY_quote | pca_1 | downside | q10_minus_q50_h6 | 2.846 | -1.366 | 5.885 |
| clock_NY_quote | pca_1 | downside | h6_minus_h1_q10 | 2.626 | -2.294 | 14.274 |
| clock_NY_quote | pca_1 | downside | h12_minus_h1_q10 | 4.576 | 0.652 | 14.108 |
| clock_NY_quote | pca_2 | account_ls | q10_minus_q50_h6 | -0.217 | -1.508 | 1.087 |
| clock_NY_quote | pca_2 | account_ls | h6_minus_h1_q10 | 0.931 | -1.463 | 2.352 |
| clock_NY_quote | pca_2 | account_ls | h12_minus_h1_q10 | -0.206 | -1.821 | 1.362 |
| clock_NY_quote | pca_2 | downside | q10_minus_q50_h6 | 2.168 | -1.191 | 4.401 |
| clock_NY_quote | pca_2 | downside | h6_minus_h1_q10 | 1.227 | -1.882 | 6.602 |
| clock_NY_quote | pca_2 | downside | h12_minus_h1_q10 | 5.319 | -0.529 | 9.409 |
| clock_NY_quote | sequential_ols | account_ls | q10_minus_q50_h6 | 0.805 | -0.972 | 2.258 |
| clock_NY_quote | sequential_ols | account_ls | h6_minus_h1_q10 | -2.279 | -5.370 | 1.581 |
| clock_NY_quote | sequential_ols | account_ls | h12_minus_h1_q10 | 0.641 | -3.063 | 2.601 |
| clock_NY_quote | sequential_ols | downside | q10_minus_q50_h6 | 2.845 | -1.558 | 5.907 |
| clock_NY_quote | sequential_ols | downside | h6_minus_h1_q10 | 3.114 | -1.834 | 13.966 |
| clock_NY_quote | sequential_ols | downside | h12_minus_h1_q10 | 4.850 | 0.593 | 14.196 |
| clock_FXKST_USNY_quote | ae_selected | account_ls | q10_minus_q50_h6 | -0.399 | -2.482 | 1.070 |
| clock_FXKST_USNY_quote | ae_selected | account_ls | h6_minus_h1_q10 | -1.428 | -3.631 | 1.958 |
| clock_FXKST_USNY_quote | ae_selected | account_ls | h12_minus_h1_q10 | 0.286 | -2.860 | 2.896 |
| clock_FXKST_USNY_quote | ae_selected | downside | q10_minus_q50_h6 | -0.519 | -2.484 | 2.314 |
| clock_FXKST_USNY_quote | ae_selected | downside | h6_minus_h1_q10 | -4.589 | -8.937 | 2.030 |
| clock_FXKST_USNY_quote | ae_selected | downside | h12_minus_h1_q10 | -3.978 | -10.322 | 2.575 |
| clock_FXKST_USNY_quote | pca_1 | account_ls | q10_minus_q50_h6 | -0.861 | -2.355 | 0.645 |
| clock_FXKST_USNY_quote | pca_1 | account_ls | h6_minus_h1_q10 | 0.189 | -2.803 | 2.776 |
| clock_FXKST_USNY_quote | pca_1 | account_ls | h12_minus_h1_q10 | 2.160 | -0.750 | 5.282 |
| clock_FXKST_USNY_quote | pca_1 | downside | q10_minus_q50_h6 | 1.095 | -2.041 | 4.038 |
| clock_FXKST_USNY_quote | pca_1 | downside | h6_minus_h1_q10 | -6.453 | -12.347 | 0.002 |
| clock_FXKST_USNY_quote | pca_1 | downside | h12_minus_h1_q10 | -6.678 | -12.920 | 1.716 |
| clock_FXKST_USNY_quote | pca_2 | account_ls | q10_minus_q50_h6 | -0.034 | -1.736 | 0.927 |
| clock_FXKST_USNY_quote | pca_2 | account_ls | h6_minus_h1_q10 | -0.203 | -3.968 | 1.910 |
| clock_FXKST_USNY_quote | pca_2 | account_ls | h12_minus_h1_q10 | 0.896 | -1.481 | 2.321 |
| clock_FXKST_USNY_quote | pca_2 | downside | q10_minus_q50_h6 | 0.392 | -0.800 | 3.296 |
| clock_FXKST_USNY_quote | pca_2 | downside | h6_minus_h1_q10 | -2.364 | -8.158 | 3.271 |
| clock_FXKST_USNY_quote | pca_2 | downside | h12_minus_h1_q10 | -2.442 | -5.750 | 1.684 |
| clock_FXKST_USNY_quote | sequential_ols | account_ls | q10_minus_q50_h6 | -1.014 | -2.402 | 0.596 |
| clock_FXKST_USNY_quote | sequential_ols | account_ls | h6_minus_h1_q10 | 0.156 | -2.817 | 2.782 |
| clock_FXKST_USNY_quote | sequential_ols | account_ls | h12_minus_h1_q10 | 2.182 | -0.691 | 5.264 |
| clock_FXKST_USNY_quote | sequential_ols | downside | q10_minus_q50_h6 | 1.401 | -2.054 | 3.990 |
| clock_FXKST_USNY_quote | sequential_ols | downside | h6_minus_h1_q10 | -6.341 | -12.336 | -0.054 |
| clock_FXKST_USNY_quote | sequential_ols | downside | h12_minus_h1_q10 | -6.858 | -13.074 | 1.966 |
| clock_NY_macro_end1h_quote | ae_selected | account_ls | q10_minus_q50_h6 | -0.145 | -1.832 | 1.631 |
| clock_NY_macro_end1h_quote | ae_selected | account_ls | h6_minus_h1_q10 | -0.780 | -4.002 | 1.158 |
| clock_NY_macro_end1h_quote | ae_selected | account_ls | h12_minus_h1_q10 | 0.191 | -2.525 | 2.849 |
| clock_NY_macro_end1h_quote | ae_selected | downside | q10_minus_q50_h6 | 1.920 | -2.157 | 3.683 |
| clock_NY_macro_end1h_quote | ae_selected | downside | h6_minus_h1_q10 | 1.138 | -2.170 | 9.754 |
| clock_NY_macro_end1h_quote | ae_selected | downside | h12_minus_h1_q10 | 2.421 | -2.906 | 7.809 |
| clock_NY_macro_end1h_quote | pca_1 | account_ls | q10_minus_q50_h6 | 1.013 | -0.990 | 2.382 |
| clock_NY_macro_end1h_quote | pca_1 | account_ls | h6_minus_h1_q10 | -1.951 | -5.717 | 1.498 |
| clock_NY_macro_end1h_quote | pca_1 | account_ls | h12_minus_h1_q10 | 1.838 | -1.527 | 4.397 |
| clock_NY_macro_end1h_quote | pca_1 | downside | q10_minus_q50_h6 | 2.944 | -1.881 | 6.457 |
| clock_NY_macro_end1h_quote | pca_1 | downside | h6_minus_h1_q10 | 2.883 | -1.276 | 12.132 |
| clock_NY_macro_end1h_quote | pca_1 | downside | h12_minus_h1_q10 | 3.347 | -1.678 | 7.581 |
| clock_NY_macro_end1h_quote | pca_2 | account_ls | q10_minus_q50_h6 | -0.062 | -1.102 | 1.194 |
| clock_NY_macro_end1h_quote | pca_2 | account_ls | h6_minus_h1_q10 | -0.847 | -2.820 | 1.495 |
| clock_NY_macro_end1h_quote | pca_2 | account_ls | h12_minus_h1_q10 | -1.056 | -2.873 | 0.683 |
| clock_NY_macro_end1h_quote | pca_2 | downside | q10_minus_q50_h6 | 1.813 | -1.983 | 4.906 |
| clock_NY_macro_end1h_quote | pca_2 | downside | h6_minus_h1_q10 | 3.855 | -2.196 | 7.955 |
| clock_NY_macro_end1h_quote | pca_2 | downside | h12_minus_h1_q10 | 4.459 | 0.670 | 8.222 |
| clock_NY_macro_end1h_quote | sequential_ols | account_ls | q10_minus_q50_h6 | 1.057 | -1.012 | 2.400 |
| clock_NY_macro_end1h_quote | sequential_ols | account_ls | h6_minus_h1_q10 | -1.896 | -5.803 | 1.338 |
| clock_NY_macro_end1h_quote | sequential_ols | account_ls | h12_minus_h1_q10 | 1.924 | -1.586 | 4.274 |
| clock_NY_macro_end1h_quote | sequential_ols | downside | q10_minus_q50_h6 | 2.901 | -1.522 | 6.686 |
| clock_NY_macro_end1h_quote | sequential_ols | downside | h6_minus_h1_q10 | 2.852 | -1.408 | 11.867 |
| clock_NY_macro_end1h_quote | sequential_ols | downside | h12_minus_h1_q10 | 3.310 | -1.666 | 7.905 |

q10−q50의 downside가 안정적으로 음수여야 중앙부보다 하방에서 더 큰 음의 관련성이라는 해석이 가능하다. h6−h1/h12−h1은 같은 origin 표본 비교다. 유의하지 않다는 결과는 효과가 정확히 0임을 입증하지 않는다.
