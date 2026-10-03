# Báo cáo Thực hành Lab 16: Cloud AI Environment Setup (Track 2)

1. **Hạ tầng & Cấu hình:** Tôi dùng **AWS**, Region **us-east-1** (us-east-1a, us-east-1b), Instance type: Compute Node **`t3.medium`** (2 vCPU, 4 GiB RAM) và Bastion Host **`t3.micro`** (1 vCPU, 1 GiB RAM), source commit **`55539f67d7c78b43afe334a2ec3271c4bfdbbe2d`** (55539f6).

2. **Dữ liệu & Phân chia:** Dataset Credit Card Fraud Detection có **284,807 dòng** (31 cột, trong đó 284,315 nhãn 0 và 492 nhãn 1, tỷ lệ gian lận 0.173%), chia train/test theo tỷ lệ **80/20** (Train: 227,845 dòng, Test: 56,962 dòng) có phân tầng theo nhãn (`stratify=y`) để bảo toàn tỷ lệ mất cân bằng, seed `random_state=42`.

3. **Thời gian huấn luyện:** Load dữ liệu mất **2.4310 giây**; training LightGBM (`class_weight='balanced'`, `n_estimators=500`, `n_jobs=-1`) mất **15.3405 giây**; best iteration là **500**.
   - Ảnh đính kèm: `screenshots/BENCHMARK_screenshots.png`.

4. **Chỉ số đánh giá (Evaluation Metrics):** Với ngưỡng phân loại tối ưu (optimal threshold) = 0.91, mô hình đạt **AUC-ROC 0.980240**, **Accuracy 0.999561** (99.96%), **F1-Score 0.866310**, **Precision 0.910112** (91.01%), **Recall 0.826531** (82.65%) trên tập test (Confusion matrix: TN=56,856, FP=8, FN=17, TP=81).

5. **Tốc độ suy luận (Inference):** Latency 1 dòng đạt **1.4274 ms**; throughput batch 1.000 dòng đạt **55,770 dòng/giây** (thời gian xử lý batch 1.000 dòng là 17.9307 ms); cách đo: thực hiện sau 10 lần warm-up, latency đo bằng giá trị trung vị (median) của 100 lần predict liên tiếp bằng `predict_proba()`.

6. **Tài nguyên hệ thống:** Ảnh chụp màn hình được ghi nhận **ngay sau khi benchmark hoàn tất**:
   - **CPU:** CPU đã hạ nhiệt về mức nghỉ với `99.7% idle`, `0.0% user`; chỉ số `load average: 0.13, 0.12, 0.04` thể hiện dấu vết tải từ tiến trình training vừa kết thúc trước đó.
   - **RAM:** Sau khi tiến trình Python giải phóng bộ nhớ, RAM used chỉ còn `241 MiB` (`229.7 MiB` trên `top`), `free 1.6 GiB`, `buff/cache 1.9 GiB` (Linux cache lại dữ liệu dataset vào bộ nhớ đệm), và `available 3.2 GiB`.
   - **Network:** Interface `ens5` ghi nhận tổng lưu lượng tích lũy sau khi hoàn thành tải môi trường và dataset: RX đạt `337 MB` (337,091,726 bytes, 255,240 packets) và TX đạt `3.6 MB` (3,684,765 bytes, 38,007 packets).
   - Ảnh đính kèm: `screenshots/CPU_screenshots.png`, `screenshots/RAM_screenshots.png`, `screenshots/NETWORK_screenshots.png`.

7. **Chi phí & Billing:** Billing tại **AWS Billing and Cost Management** (Tài khoản `972243443873` - `AI-Lab-Group`) ghi nhận tổng chi phí lũy kế đầu tháng đến hiện tại (Month-to-date cost) là **$2.84 USD**. Chi tiết phân bổ chi phí các dịch vụ liên quan:
   - **EC2 - Other** (gồm NAT Gateway, EBS Volume, Elastic IP): `$1.02 USD`
   - **Amazon Elastic Compute Cloud - Compute** (máy ảo EC2 `t3.medium`, `t3.micro`): `$0.94 USD`
   - **Amazon Elastic Load Balancing** (Application Load Balancer - ALB): `$0.47 USD`
   - **Amazon Virtual Private Cloud** (VPC): `$0.41 USD`
   - Ảnh đính kèm: `screenshots/bill_screenshots.png`.

8. **Thu dọn tài nguyên:** Tôi đã tải toàn bộ kết quả benchmark và xóa tài nguyên lúc **02:23 AM (GMT+7) ngày 04/10/2026**; bằng chứng dọn dẹp: lệnh `terraform destroy -auto-approve` trả về **`Destroy complete! Resources: 27 destroyed.`** và lệnh kiểm tra `terraform state list` trả về kết quả rỗng (0 resource).
   - Ảnh đính kèm: `screenshots/Destroy_screenshots.png`.
