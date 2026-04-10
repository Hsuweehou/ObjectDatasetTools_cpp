#include "odt/utils/async_progress.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <thread>

namespace odt {

namespace {
constexpr int kBarWidth = 28;
constexpr int kClearWidth = 96;
}  // namespace

AsyncProgress::AsyncProgress(std::ostream& out) : out_(out) {
    display_thread_ = std::thread([this]() { display_loop(); });
}

AsyncProgress::~AsyncProgress() {
    alive_.store(false, std::memory_order_release);
    if (display_thread_.joinable()) {
        display_thread_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    clear_line_unlocked();
}

void AsyncProgress::begin_step(std::string name, int64_t total) {
    std::lock_guard<std::mutex> lock(mutex_);
    step_name_ = std::move(name);
    total_.store(std::max<int64_t>(total, 0), std::memory_order_relaxed);
    current_.store(0, std::memory_order_relaxed);
    indeterminate_.store(total <= 0, std::memory_order_relaxed);
}

void AsyncProgress::set_current(int64_t value) {
    current_.store(std::max<int64_t>(value, 0), std::memory_order_relaxed);
}

void AsyncProgress::advance(int64_t delta) {
    current_.fetch_add(delta, std::memory_order_relaxed);
}

void AsyncProgress::end_step() {
    std::lock_guard<std::mutex> lock(mutex_);
    clear_line_unlocked();
    step_name_.clear();
    indeterminate_.store(false, std::memory_order_relaxed);
    total_.store(1, std::memory_order_relaxed);
    current_.store(0, std::memory_order_relaxed);
}

void AsyncProgress::clear_line_unlocked() {
    out_ << '\r' << std::string(static_cast<size_t>(kClearWidth), ' ') << '\r'
         << std::flush;
}

void AsyncProgress::display_loop() {
    using namespace std::chrono_literals;
    int spin = 0;
    const char spin_chars[] = "|/-\\";
    while (alive_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(80ms);
        std::string name;
        int64_t cur = 0;
        int64_t tot = 1;
        bool indet = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!alive_.load(std::memory_order_relaxed)) {
                break;
            }
            name = step_name_;
            cur = current_.load(std::memory_order_relaxed);
            tot = total_.load(std::memory_order_relaxed);
            indet = indeterminate_.load(std::memory_order_relaxed);
        }
        if (name.empty()) {
            continue;
        }
        std::ostringstream line;
        line << "\r[odt] ";
        if (indet || tot <= 0) {
            line << spin_chars[spin % 4] << "  " << name;
            ++spin;
        } else {
            const int64_t denom = std::max<int64_t>(tot, 1);
            const int pct = static_cast<int>(
                std::min<int64_t>(100, cur * 100 / denom));
            const int filled = static_cast<int>(cur * kBarWidth / denom);
            line << '[';
            for (int i = 0; i < kBarWidth; ++i) {
                line << (i < filled ? '#' : '.');
            }
            line << "] " << std::setw(3) << pct << "%  " << name;
        }
        out_ << line.str() << std::flush;
    }
}

}  // namespace odt
