Đây là task breakdown rõ hơn cho cả team:

***

## 🚀 Bắt đầu (Tất cả)

1. Pull code về, tạo conda env
2. Cài thư viện — xem `README.md` phần Requirements
3. Đọc `REPORT.md` trước, rồi `README.md`
4. Chạy thử 3 terminal trên 1 máy theo hướng dẫn README
5. Nếu chạy được → hẹn ngày chạy thật 3 máy qua LAN

***

## 🤖 Team AI — Tèo & Khôi

**Mục tiêu: làm cho model xịn hơn**

- [ ] Xem lại `local_trainer.py` — model hiện tại train quá ít (`max_steps=3`), cần bỏ giới hạn đó đi
- [ ] Xem lại `inference.py` — prompt format phải khớp với training, đọc bug #3 trong Report
- [ ] Xem lại `aggregator.py` — đọc phần FedIT SVD trong Report để hiểu tại sao không dùng naive average
- [ ] Split dataset thành 3 file `part1.jsonl`, `part2.jsonl`, `part3.jsonl` — hiện tại dataset đang fake
- [ ] Thêm GPU support cho máy Hóa dùng Intel ARC — cần flag `device="xpu"` hoặc kiểm tra `torch.xpu.is_available()`
- [ ] Evaluation graph — không bắt buộc, làm nếu còn thời gian

***

## 📡 Team Communication — Hoàng & Hóa & Minh

**Mục tiêu: đảm bảo communication layer không có lỗ hổng**

- [ ] Đọc kỹ phần 4 và 5 trong `REPORT.md` (Communication + Polling Barrier)
- [ ] Tìm edge case còn thiếu — gợi ý bên dưới
- [ ] Stress test: chạy 3 node, kill 1 node giữa chừng, xem còn 2 node có tiếp tục không
- [ ] Hoàng: nghĩ cách làm **central dashboard** — một trang web xem status của cả 3 node cùng lúc (poll `/status` của từng node rồi render ra)

### Edge cases cần check

| Case | Hiện tại xử lý chưa? |
|---|---|
| 1 node crash giữa training | Poll timeout → skip, cluster tiếp tục ✅ |
| 1 node crash giữa aggregation | Peer nhận 404 trên `/weights` → skip ✅ |
| Node chậm hơn timeout 10 phút | Bị skip round đó ✅ |
| Node reconnect sau khi chết | ❌ Chưa có |
| 2/3 node crash cùng lúc | ❌ Chưa test |
| Dataset của 1 node rỗng | ❌ Chưa handle |

***

## 💭 Chủ đề Hóa đang nghĩ — Heartbeat & Reconnect

Đây là 2 câu hỏi thiết kế hay, tóm tắt lại cho rõ:

### Heartbeat
Ý tưởng: mỗi node định kỳ ping các peer, nếu peer không trả lời trong X giây thì coi là "chết". Hiện tại cơ chế poll `/status` đã làm việc này một phần — nếu peer không trả lời trong `poll_timeout` thì bị skip. Câu hỏi là có cần **chủ động** hơn không (ping liên tục ngay cả ngoài round).

### Reconnect sau khi chết
Đây là vấn đề khó vì hệ thống có biến `round`. Phân tích:

- Nếu node chết ở round 2 rồi reconnect, nó đang ở round 1 trong khi các node khác đang round 3 → không thể merge
- **Hướng giải quyết có thể:** node reconnect tự nhận adapter mới nhất từ một peer (GET `/weights` của round hiện tại) rồi "nhảy" lên round đó, bỏ qua các round đã miss
- Hóa đúng: merge 3 node dù khác thời gian vẫn tốt hơn merge 2 — đây là lý do worth implement

Đây là feature nâng cao, nên để sau khi hệ thống cơ bản chạy ổn trên LAN đã.