#include "odt/cli.hpp"

#include <cstring>
#include <iostream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

/// 与源码 UTF-8（/utf-8）一致，避免 Windows 控制台默认 GBK 下中文乱码
static void InitConsoleUtf8() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

static void PrintUsage(const char* exe) {
    std::cerr
        << "Usage: " << exe
        << " <command> [args...]\n"
        << "Commands:\n"
        << "  record_air           <output_dir> [timed ... | interactive ...]\n"
        << "  compute_gt_poses     <sequence_dir>\n"
        << "  register_scene       <path|all>\n"
        << "  create_label_files   <path|all>\n";
}

int main(int argc, char** argv) {
    InitConsoleUtf8();
    if (argc < 2) {
        PrintUsage(argc >= 1 ? argv[0] : "object_dataset_tools");
        return 1;
    }
    const char* cmd = argv[1]; // record_air D:\codes\project_template_scanner\algorithm\ObjectDatasetTools_cpp\data\cylinder_bottle1 interactive 10.30.7.157 50051 D:\codes\project_template_scanner\algorithm\ObjectDatasetTools_cpp\config\scanner0_params.json
    if (std::strcmp(cmd, "record_air") == 0) {
        return odt::RunRecordAir(argc - 1, argv + 1);
    }
    if (std::strcmp(cmd, "compute_gt_poses") == 0) { // compute_gt_poses D:\codes\project_template_scanner\algorithm\ObjectDatasetTools_cpp\data\cylinder_bottle1
        return odt::RunComputeGtPoses(argc - 1, argv + 1);
    }
    if (std::strcmp(cmd, "register_scene") == 0) { // register_scene D:\codes\project_template_scanner\algorithm\ObjectDatasetTools_cpp\data\cylinder_bottle1
        return odt::RunRegisterScene(argc - 1, argv + 1);
    }
    if (std::strcmp(cmd, "create_label_files") == 0) {
        return odt::RunCreateLabelFiles(argc - 1, argv + 1);
    }
    std::cerr << "Unknown command: " << cmd << "\n";
    PrintUsage(argv[0]);


    return 1;
}
