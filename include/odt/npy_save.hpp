#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace odt {

/// 写入 float64、C 连续、shape (n0, n1, n2) 的 .npy（与 numpy.save 兼容）
inline bool SaveNpyFloat64_3D(const std::string& path,
                              const double* data,
                              size_t n0,
                              size_t n1,
                              size_t n2) {
    std::ostringstream hdr;
    hdr << "{'descr': '<f8', 'fortran_order': False, 'shape': (" << n0 << ", "
        << n1 << ", " << n2 << "), }";
    std::string header = hdr.str();
    const size_t header_len = header.size() + 1;
    const size_t padding =
        (64 - ((10 + header_len) % 64)) % 64;
    header.append(padding, ' ');
    header.push_back('\n');

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    static const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    out.write(magic, 6);
    char ver[2] = {1, 0};
    out.write(ver, 2);
    uint16_t hl = static_cast<uint16_t>(header.size());
    out.write(reinterpret_cast<const char*>(&hl), 2);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    const size_t count = n0 * n1 * n2;
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(count * sizeof(double)));
    return true;
}

/// 与 numpy.save 的二维 float64 数组一致，例如 shape (4, 4)
inline bool SaveNpyFloat64_2D(const std::string& path,
                              const double* data,
                              size_t n0,
                              size_t n1) {
    std::ostringstream hdr;
    hdr << "{'descr': '<f8', 'fortran_order': False, 'shape': (" << n0 << ", "
        << n1 << "), }";
    std::string header = hdr.str();
    const size_t header_len = header.size() + 1;
    const size_t padding =
        (64 - ((10 + header_len) % 64)) % 64;
    header.append(padding, ' ');
    header.push_back('\n');

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    static const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    out.write(magic, 6);
    char ver[2] = {1, 0};
    out.write(ver, 2);
    uint16_t hl = static_cast<uint16_t>(header.size());
    out.write(reinterpret_cast<const char*>(&hl), 2);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    const size_t count = n0 * n1;
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(count * sizeof(double)));
    return true;
}

}  // namespace odt
