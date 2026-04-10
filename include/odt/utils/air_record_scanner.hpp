#pragma once

#include <AIRScanner.h>

#include <cstdint>
#include <string>

namespace odt {

/// 与采集脚本 record_air 相关的网络相机连接参数
struct AirScannerConfig {
    std::string   ip                 = "10.30.7.157";
    std::uint16_t port               = 50051;
    unsigned int  connect_timeout_ms = 3000;
    /// AIR_ImportConfig 使用的参数 JSON（与 scanner_a 的 scanner_param_path 一致）；须先导入再 GetCameraParams 才有有效内参
    std::string   scanner_param_path;
};

/// 仅封装 AIR C API 中 record_air 所需能力（Initialize / Connect / ImportConfig / 取参 / Capture / Release）
class AirRecordScanner {
public:
    explicit AirRecordScanner(const AirScannerConfig& config = {});
    ~AirRecordScanner();

    AirRecordScanner(const AirRecordScanner&)            = delete;
    AirRecordScanner& operator=(const AirRecordScanner&) = delete;

    bool Initialize();
    bool Connect();
    bool Disconnect();

    bool IsConnected() const;

    /// 从 JSON 导入相机标定/参数（与 scanner_a::Scanner::ImportConfig 一致）；@param file_path 为空则用 config_.scanner_param_path
    bool ImportConfig(const std::string& file_path = "");

    bool CaptureFrame(AIRFrameData& frame_data);
    bool ReleaseFrame(AIRFrameData& frame_data);

    bool GetCameraParams(AIRScannerParams& out_params);

    AIRError GetLastError() const { return last_error_; }

    void PrintLastError() const;

    AIRScannerHandle        GetHandle() const { return handle_; }
    const AirScannerConfig& GetConfig() const { return config_; }

private:
    AirScannerConfig   config_;
    AIRScannerHandle   handle_{nullptr};
    bool               is_initialized_{false};
    bool               is_connected_{false};
    AIRError           last_error_{};
    mutable std::string last_error_message_;
    mutable std::string last_error_file_;
    mutable std::string last_error_function_;

    AIRError CreateError(AIRstatus status,
                         const char* message,
                         const char* file     = nullptr,
                         const char* function = nullptr,
                         int         line     = 0) const;
    bool     IsHandleValid() const;
};

}  // namespace odt
