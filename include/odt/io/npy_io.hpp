#pragma once

/// .npy 文件与 Python np.save 互操作；数组数据为 NumCpp NdArray<double>
///（原 `npy_save.hpp` 中的 SaveNpyFloat64_* 已合并至本文件）
/// 勿包含 <NumCpp.hpp>：会拉取 DateTime 等模块并依赖 Boost；reshape.hpp 已依赖 NdArray

#include "NumCpp/Functions/reshape.hpp"

#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

namespace odt {
namespace detail {

inline bool parse_npy_header_shape3(const std::string& header,
                                    std::size_t&       n0,
                                    std::size_t&       n1,
                                    std::size_t&       n2) {
    const std::string key = "shape";
    std::size_t       p   = header.find(key);
    if(p == std::string::npos) {
        return false;
    }
    p = header.find('(', p);
    if(p == std::string::npos) {
        return false;
    }
    const std::size_t q = header.find(')', p);
    if(q == std::string::npos || q <= p + 1) {
        return false;
    }
    std::string              inner = header.substr(p + 1, q - p - 1);
    std::vector<std::size_t> dims;
    std::stringstream        ss(inner);
    std::string              part;
    while(std::getline(ss, part, ',')) {
        while(!part.empty() && (part.front() == ' ' || part.front() == '\n' || part.front() == '\r')) {
            part.erase(part.begin());
        }
        while(!part.empty() && (part.back() == ' ' || part.back() == '\n' || part.back() == '\r')) {
            part.pop_back();
        }
        if(part.empty()) {
            continue;
        }
        try {
            dims.push_back(static_cast<std::size_t>(std::stoull(part)));
        } catch(...) {
            return false;
        }
    }
    if(dims.size() != 3) {
        return false;
    }
    n0 = dims[0];
    n1 = dims[1];
    n2 = dims[2];
    return true;
}

inline bool write_npy_f64(const std::string& path, const double* data, std::size_t num_elements,
                          std::initializer_list<std::size_t> shape) {
    std::ostringstream hdr;
    hdr << "{'descr': '<f8', 'fortran_order': False, 'shape': (";
    bool first = true;
    for(const auto d : shape) {
        if(!first) {
            hdr << ", ";
        }
        first = false;
        hdr << d;
    }
    hdr << "), }";
    std::string       header    = hdr.str();
    const std::size_t header_len = header.size() + 1;
    const std::size_t padding    = (64 - ((10 + header_len) % 64)) % 64;
    header.append(padding, ' ');
    header.push_back('\n');

    std::ofstream out(path, std::ios::binary);
    if(!out) {
        return false;
    }
    static const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    out.write(magic, 6);
    char ver[2] = {1, 0};
    out.write(ver, 2);
    const auto hl = static_cast<std::uint16_t>(header.size());
    out.write(reinterpret_cast<const char*>(&hl), 2);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(num_elements * sizeof(double)));
    return static_cast<bool>(out);
}

}  // namespace detail

/// 读取 .npy（float64，逻辑三维）；\p out  reshape 为 (d0, d1*d2)
inline bool load_ndarray_npy(const std::string& path, nc::NdArray<double>& out, std::size_t& d0,
                             std::size_t& d1, std::size_t& d2) {
    std::ifstream in(path, std::ios::binary);
    if(!in) {
        return false;
    }
    char magic[6];
    in.read(magic, 6);
    if(in.gcount() != 6 || magic[0] != '\x93' || std::string(magic + 1, magic + 6) != "NUMPY") {
        return false;
    }
    std::uint8_t ver_maj = 0, ver_min = 0;
    in.read(reinterpret_cast<char*>(&ver_maj), 1);
    in.read(reinterpret_cast<char*>(&ver_min), 1);
    std::uint16_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(static_cast<std::size_t>(header_len), '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_len));
    if(!detail::parse_npy_header_shape3(header, d0, d1, d2)) {
        return false;
    }
    const std::size_t count = d0 * d1 * d2;
    if(count == 0) {
        out = nc::NdArray<double>();
        return true;
    }
    out = nc::NdArray<double>(static_cast<nc::uint32>(count), 1u);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(double)));
    if(!in || static_cast<std::size_t>(in.gcount()) != count * sizeof(double)) {
        return false;
    }
    nc::reshape(out, static_cast<nc::int32>(d0), static_cast<nc::int32>(d1 * d2));
    return true;
}

inline bool save_ndarray_npy(const std::string& path, const nc::NdArray<double>& arr, std::size_t d0,
                             std::size_t d1, std::size_t d2) {
    if(static_cast<std::size_t>(arr.size()) != d0 * d1 * d2) {
        return false;
    }
    return detail::write_npy_f64(path, arr.data(), d0 * d1 * d2, {d0, d1, d2});
}

inline bool save_ndarray_npy(const std::string& path, const nc::NdArray<double>& arr, std::size_t d0,
                             std::size_t d1) {
    if(static_cast<std::size_t>(arr.size()) != d0 * d1) {
        return false;
    }
    return detail::write_npy_f64(path, arr.data(), d0 * d1, {d0, d1});
}

/// 写入 float64、C 连续、shape (n0, n1, n2) 的 .npy（与 numpy.save 兼容）
inline bool SaveNpyFloat64_3D(const std::string& path,
                              const double*     data,
                              std::size_t       n0,
                              std::size_t       n1,
                              std::size_t       n2) {
    return detail::write_npy_f64(path, data, n0 * n1 * n2, {n0, n1, n2});
}

/// 与 numpy.save 的二维 float64 数组一致，例如 shape (4, 4)
inline bool SaveNpyFloat64_2D(const std::string& path,
                              const double*     data,
                              std::size_t       n0,
                              std::size_t       n1) {
    return detail::write_npy_f64(path, data, n0 * n1, {n0, n1});
}

}  // namespace odt
