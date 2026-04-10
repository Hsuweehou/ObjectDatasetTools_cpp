#pragma once

#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>

namespace odt {

/// 在独立线程中绘制终端进度条；业务线程只更新原子计数，不因刷新进度而阻塞在 I/O。
class AsyncProgress {
public:
    explicit AsyncProgress(std::ostream& out = std::cerr);
    ~AsyncProgress();

    AsyncProgress(const AsyncProgress&) = delete;
    AsyncProgress& operator=(const AsyncProgress&) = delete;

    /// total<=0 时为不确定进度（旋转指示）
    void begin_step(std::string name, int64_t total);
    void set_current(int64_t value);
    void advance(int64_t delta = 1);
    /// 清行并换行，便于后续 std::cout 输出
    void end_step();

private:
    void display_loop();
    void clear_line_unlocked();

    std::ostream& out_;
    std::mutex mutex_;
    std::string step_name_;
    std::thread display_thread_;
    std::atomic<int64_t> current_{0};
    std::atomic<int64_t> total_{1};
    std::atomic<bool> indeterminate_{false};
    std::atomic<bool> alive_{true};
};

}  // namespace odt
