#include "odt/air_record_scanner.hpp"

#include <cstring>
#include <iostream>
#include <string>

namespace odt {

AirRecordScanner::AirRecordScanner(const AirScannerConfig& config)
    : config_(config) {
    last_error_.status   = AIRstatus::kUnknown;
    last_error_.type     = AIR_EXCEPTION_TYPE_UNKNOWN;
    last_error_.message  = nullptr;
    last_error_.file     = nullptr;
    last_error_.function = nullptr;
    last_error_.line     = 0;
}

AirRecordScanner::~AirRecordScanner() {
    Disconnect();
    if(handle_ != nullptr) {
        AIR_DestroyScanner(handle_);
        handle_ = nullptr;
    }
}

bool AirRecordScanner::Initialize() {
    if(is_initialized_) {
        last_error_ = CreateError(AIRstatus::kSuccess, "Scanner already initialized");
        return true;
    }

    handle_ = AIR_CreateScanner();
    if(handle_ == nullptr) {
        last_error_ = CreateError(AIRstatus::kFail, "Failed to create scanner handle", __FILE__, __FUNCTION__, __LINE__);
        PrintLastError();
        return false;
    }

    if(!AIR_IsValidHandle(handle_)) {
        last_error_ = CreateError(AIRstatus::kFail, "Invalid scanner handle", __FILE__, __FUNCTION__, __LINE__);
        PrintLastError();
        AIR_DestroyScanner(handle_);
        handle_ = nullptr;
        return false;
    }

    is_initialized_ = true;
    last_error_     = CreateError(AIRstatus::kSuccess, "Scanner initialized successfully");
    return true;
}

bool AirRecordScanner::Connect() {
    if(!is_initialized_ && !Initialize()) {
        return false;
    }

    if(is_connected_) {
        last_error_ = CreateError(AIRstatus::kSuccess, "Already connected");
        return true;
    }

    AIRScannerInfo info{};
    std::strncpy(info.ip, config_.ip.c_str(), AIR_IP_MAX - 1);
    info.ip[AIR_IP_MAX - 1] = '\0';
    info.port               = config_.port;

    AIRstatus status = AIR_Connect(handle_, &info, config_.connect_timeout_ms);

    if(status != AIRstatus::kSuccess) {
        const std::string error_msg =
            std::string("Failed to connect to scanner at ") + info.ip + ":" + std::to_string(info.port);
        last_error_ = CreateError(status, error_msg.c_str(), __FILE__, __FUNCTION__, __LINE__);
        PrintLastError();
        return false;
    }

    is_connected_ = true;
    last_error_   = CreateError(AIRstatus::kSuccess, "Connected to scanner successfully");
    return true;
}

bool AirRecordScanner::Disconnect() {
    if(!is_connected_) {
        return true;
    }

    if(IsHandleValid()) {
        AIRstatus status = AIR_Disconnect(handle_);
        if(status == AIRstatus::kSuccess) {
            last_error_ = CreateError(AIRstatus::kSuccess, "Disconnected from scanner");
        } else {
            last_error_ = CreateError(status, "Failed to disconnect from scanner", __FILE__, __FUNCTION__, __LINE__);
        }
    }

    is_connected_ = false;
    return true;
}

bool AirRecordScanner::IsConnected() const {
    if(!IsHandleValid()) {
        return false;
    }
    return is_connected_ && AIR_IsConnected(handle_);
}

bool AirRecordScanner::ImportConfig(const std::string& file_path) {
    if(!IsConnected()) {
        last_error_ = CreateError(AIRstatus::kNotConnected,
                                 "Scanner is not connected. Cannot import config.",
                                 __FILE__,
                                 __FUNCTION__,
                                 __LINE__);
        PrintLastError();
        return false;
    }

    const std::string config_file = file_path.empty() ? config_.scanner_param_path : file_path;
    if(config_file.empty()) {
        last_error_ = CreateError(AIRstatus::kInvalidParameter,
                                 "scanner_param_path is empty; set AirScannerConfig::scanner_param_path or pass ImportConfig(path).",
                                 __FILE__,
                                 __FUNCTION__,
                                 __LINE__);
        PrintLastError();
        return false;
    }

    const AIRstatus status = AIR_ImportConfig(handle_, config_file.c_str());
    if(status != AIRstatus::kSuccess) {
        const std::string error_msg = std::string("Failed to import config from: ") + config_file;
        last_error_                 = CreateError(status, error_msg.c_str(), __FILE__, __FUNCTION__, __LINE__);
        PrintLastError();
        return false;
    }

    last_error_ = CreateError(AIRstatus::kSuccess, "Configuration imported successfully");
    return true;
}

bool AirRecordScanner::CaptureFrame(AIRFrameData& frame_data) {
    if(!IsConnected()) {
        last_error_ = CreateError(AIRstatus::kNotConnected, "Scanner is not connected", __FILE__, __FUNCTION__, __LINE__);
        PrintLastError();
        return false;
    }

    AIRstatus status = AIR_Capture(handle_, &frame_data);

    if(status != AIRstatus::kSuccess) {
        last_error_ = CreateError(status, "Failed to capture frame", __FILE__, __FUNCTION__, __LINE__);
        PrintLastError();
        return false;
    }

    last_error_ = CreateError(AIRstatus::kSuccess, "Frame captured successfully");
    return true;
}

bool AirRecordScanner::ReleaseFrame(AIRFrameData& frame_data) {
    if(!IsHandleValid()) {
        return false;
    }

    AIRstatus status = AIR_ReleaseFrame(handle_, &frame_data);
    if(status == AIRstatus::kSuccess) {
        last_error_ = CreateError(AIRstatus::kSuccess, "Frame resources released");
    } else {
        last_error_ = CreateError(status, "Failed to release frame resources", __FILE__, __FUNCTION__, __LINE__);
    }
    return status == AIRstatus::kSuccess;
}

bool AirRecordScanner::GetCameraParams(AIRScannerParams& out_params) {
    if(!IsConnected()) {
        last_error_ = CreateError(AIRstatus::kNotConnected,
                                 "Scanner is not connected. Cannot get camera parameters.",
                                 __FILE__,
                                 __FUNCTION__,
                                 __LINE__);
        PrintLastError();
        return false;
    }

    AIRstatus status = AIR_GetScannerParams(handle_, &out_params);

    if(status != AIRstatus::kSuccess) {
        last_error_ = CreateError(status, "Failed to get camera parameters", __FILE__, __FUNCTION__, __LINE__);
        PrintLastError();
        return false;
    }

    last_error_ = CreateError(AIRstatus::kSuccess, "Camera parameters retrieved successfully");
    return true;
}

bool AirRecordScanner::IsHandleValid() const {
    return handle_ != nullptr && AIR_IsValidHandle(handle_);
}

AIRError AirRecordScanner::CreateError(AIRstatus status,
                                       const char* message,
                                       const char* file,
                                       const char* function,
                                       int         line) const {
    AIRError error{};
    error.status = status;
    error.line   = line;

    if(message) {
        last_error_message_ = message;
        error.message       = last_error_message_.c_str();
    } else {
        error.message = nullptr;
    }

    if(file) {
        last_error_file_ = file;
        error.file       = last_error_file_.c_str();
    } else {
        error.file = nullptr;
    }

    if(function) {
        last_error_function_ = function;
        error.function       = last_error_function_.c_str();
    } else {
        error.function = nullptr;
    }

    switch(status) {
    case AIRstatus::kInvalidParameter:
        error.type = AIR_EXCEPTION_TYPE_INVALID_ARGUMENT;
        break;
    case AIRstatus::kNotConnected:
        error.type = AIR_EXCEPTION_TYPE_IO;
        break;
    case AIRstatus::kNotSupported:
        error.type = AIR_EXCEPTION_TYPE_NOT_SUPPORTED;
        break;
    case AIRstatus::kBusy:
        error.type = AIR_EXCEPTION_TYPE_RUNTIME;
        break;
    case AIRstatus::kFail:
        error.type = AIR_EXCEPTION_TYPE_RUNTIME;
        break;
    case AIRstatus::kSuccess:
        error.type = AIR_EXCEPTION_TYPE_UNKNOWN;
        break;
    default:
        error.type = AIR_EXCEPTION_TYPE_UNKNOWN;
        break;
    }

    return error;
}

void AirRecordScanner::PrintLastError() const {
    if(last_error_.status == AIRstatus::kSuccess) {
        if(last_error_.message) {
            std::cerr << "[AirRecordScanner] " << last_error_.message << "\n";
        }
        return;
    }

    std::cerr << "=== AirRecordScanner error ===\n";
    std::cerr << "Status: " << static_cast<int>(last_error_.status) << "\n";
    std::cerr << "Exception type: " << static_cast<int>(last_error_.type) << "\n";
    if(last_error_.message) {
        std::cerr << "Message: " << last_error_.message << "\n";
    }
    if(last_error_.file) {
        std::cerr << "File: " << last_error_.file << "\n";
    }
    if(last_error_.function) {
        std::cerr << "Function: " << last_error_.function << "\n";
    }
    if(last_error_.line > 0) {
        std::cerr << "Line: " << last_error_.line << "\n";
    }
    std::cerr << "==============================\n";
}

}  // namespace odt
