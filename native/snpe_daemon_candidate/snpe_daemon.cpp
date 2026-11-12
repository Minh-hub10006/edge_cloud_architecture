//==============================================================================
// SNPE persistent inference daemon â€” keeps DLC loaded, serves Unix socket.
// Protocol v1: see detectors/npu/snpe_daemon_client.py
//==============================================================================

#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <vector>

#include "DlContainer/DlContainer.h"
#include "DlSystem/DlEnums.h"
#include "DlSystem/ITensor.h"
#include "DlSystem/StringList.h"
#include "DlSystem/TensorMap.h"
#include "SNPE/SNPE.h"
#include "SNPE/SNPEBuilder.h"
#include "SNPE/SNPEUtil.h"

namespace {

constexpr uint32_t kMagic = 0x45504e53;  // 'SNPE' little-endian
constexpr uint8_t kProtoVersion = 1;
constexpr uint8_t kCmdPing = 1;
constexpr uint8_t kCmdInfer = 2;
constexpr uint8_t kCmdShutdown = 3;
constexpr uint8_t kCmdInferShm = 4;

std::string g_shm_dir;

bool read_exact(int fd, void* buf, size_t len) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t got = 0;
    while (got < len) {
        const ssize_t n = ::read(fd, p + got, len - got);
        if (n <= 0) return false;
        got += static_cast<size_t>(n);
    }
    return true;
}

bool write_exact(int fd, const void* buf, size_t len) {
    const auto* p = static_cast<const uint8_t*>(buf);
    size_t sent = 0;
    while (sent < len) {
        const ssize_t n = ::write(fd, p + sent, len - sent);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

Snpe_PerformanceProfile_t perf_from_string(const std::string& s) {
    if (s == "burst") return SNPE_PERFORMANCE_PROFILE_BURST;
    if (s == "high_performance") return SNPE_PERFORMANCE_PROFILE_HIGH_PERFORMANCE;
    if (s == "sustained_high_performance") return SNPE_PERFORMANCE_PROFILE_SUSTAINED_HIGH_PERFORMANCE;
    if (s == "power_saver") return SNPE_PERFORMANCE_PROFILE_POWER_SAVER;
    return SNPE_PERFORMANCE_PROFILE_BALANCED;
}

Snpe_Runtime_t runtime_from_string(const std::string& s) {
    if (s == "gpu") return SNPE_RUNTIME_GPU;
    if (s == "dsp" || s == "htp") return SNPE_RUNTIME_DSP;
    if (s == "aip") return SNPE_RUNTIME_AIP_FIXED8_TF;
    return SNPE_RUNTIME_CPU_FLOAT32;
}

Snpe_SNPE_Handle_t build_snpe(
    const std::string& dlc,
    Snpe_Runtime_t runtime,
    Snpe_PerformanceProfile_t perf,
    bool init_cache,
    const std::vector<std::string>& output_tensors,
    const std::string& graph,
    bool legacy_output_api) {
    Snpe_DlContainer_Handle_t container = Snpe_DlContainer_Open(dlc.c_str());
    if (!container) {
        std::cerr << "[snpe-daemon] failed to open DLC: " << dlc << std::endl;
        return nullptr;
    }

    Snpe_RuntimeList_Handle_t rt_list = Snpe_RuntimeList_Create();
    if (!Snpe_Util_IsRuntimeAvailable(runtime)) {
        std::cerr << "[snpe-daemon] DSP unavailable; CPU fallback forbidden" << std::endl;
        Snpe_RuntimeList_Delete(rt_list);
        Snpe_DlContainer_Delete(container);
        return nullptr;
    }
    Snpe_RuntimeList_Add(rt_list, runtime);

    Snpe_PlatformConfig_Handle_t platform = Snpe_PlatformConfig_Create();
    Snpe_SNPEBuilder_Handle_t builder = Snpe_SNPEBuilder_Create(container);
    Snpe_StringList_Handle_t outs = Snpe_StringList_Create();
    Snpe_StringList_Handle_t graphs = Snpe_StringList_Create();
    bool configured = container && rt_list && platform && builder && outs && graphs;
    auto checked = [&](const char* stage, Snpe_ErrorCode_t rc) {
        if (rc != SNPE_SUCCESS) {
            std::cerr << "[snpe-daemon] config failed stage=" << stage
                      << " code=" << static_cast<int>(rc) << " "
                      << Snpe_ErrorCode_GetLastErrorString() << std::endl;
            configured = false;
        }
    };
    for (const auto& name : output_tensors) Snpe_StringList_Append(outs, name.c_str());
    if (configured) {
        if (legacy_output_api) {
            // Build-only A/B: reproduce the old selector, never serve inference.
            checked("SetOutputTensors", Snpe_SNPEBuilder_SetOutputTensors(builder, outs));
        } else {
            Snpe_StringList_Append(graphs, graph.c_str());
            checked("NetworkInit", Snpe_SNPEBuilder_NetworkInit(builder, graphs));
            if (configured) checked("SetOutputTensorsForNetwork",
                Snpe_SNPEBuilder_SetOutputTensorsForNetwork(builder, graph.c_str(), outs));
        }
    }
    if (configured) checked("RuntimeProcessorOrder",
        Snpe_SNPEBuilder_SetRuntimeProcessorOrder(builder, rt_list));
    if (configured) checked("ITensorBuffers",
        Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(builder, false));
    if (configured) checked("PlatformConfig", Snpe_SNPEBuilder_SetPlatformConfig(builder, platform));
    if (configured) checked("InitCache", Snpe_SNPEBuilder_SetInitCacheMode(builder, init_cache));
    if (configured) checked("PerformanceProfile", Snpe_SNPEBuilder_SetPerformanceProfile(builder, perf));
    std::cerr << "[snpe-daemon] contract=v3 runtime=dsp fallback=disabled graph=" << graph
              << " init_cache=" << (init_cache ? "on" : "off")
              << " selector=" << (legacy_output_api ? "legacy" : "network") << std::endl;
    Snpe_SNPE_Handle_t snpe = configured ? Snpe_SNPEBuilder_Build(builder) : nullptr;
    // Capture the error before destroying configuration handles.
    if (!snpe && configured) {
        std::cerr << "[snpe-daemon] Build failed: "
                  << Snpe_ErrorCode_GetLastErrorString() << std::endl;
    }
    Snpe_StringList_Delete(graphs);
    Snpe_StringList_Delete(outs);

    Snpe_SNPEBuilder_Delete(builder);
    Snpe_PlatformConfig_Delete(platform);
    Snpe_RuntimeList_Delete(rt_list);
    Snpe_DlContainer_Delete(container);

    return snpe;
}

bool write_error(int fd, const std::string& msg) {
    const uint8_t status = 1;
    const uint16_t len = htons(static_cast<uint16_t>(msg.size()));
    return write_exact(fd, &status, 1) && write_exact(fd, &len, 2) &&
           write_exact(fd, msg.data(), msg.size());
}

bool write_ok_header(int fd, uint32_t infer_us, uint16_t num_outputs) {
    const uint8_t status = 0;
    const uint32_t us_be = htonl(infer_us);
    const uint16_t n_be = htons(num_outputs);
    return write_exact(fd, &status, 1) && write_exact(fd, &us_be, 4) &&
           write_exact(fd, &n_be, 2);
}

bool write_output_meta(int fd, const std::string& name_str, uint32_t data_len) {
    const uint16_t name_len = htons(static_cast<uint16_t>(name_str.size()));
    const uint32_t len_be = htonl(data_len);
    return write_exact(fd, &name_len, 2) && write_exact(fd, name_str.data(), name_str.size()) &&
           write_exact(fd, &len_be, 4);
}

bool execute_infer(
    Snpe_SNPE_Handle_t snpe,
    const float* input,
    size_t input_floats,
    int fd,
    bool shm_outputs) {
    Snpe_StringList_Handle_t input_names = Snpe_SNPE_GetInputTensorNames(snpe);
    if (!input_names || Snpe_StringList_Size(input_names) == 0) {
        Snpe_StringList_Delete(input_names);
        return write_error(fd, "no input tensor names");
    }

    const char* input_name = Snpe_StringList_At(input_names, 0);
    Snpe_TensorShape_Handle_t shape = Snpe_SNPE_GetInputDimensions(snpe, input_name);
    if (!shape) {
        Snpe_StringList_Delete(input_names);
        return write_error(fd, "input dimensions unavailable");
    }

    Snpe_ITensor_Handle_t itensor = Snpe_Util_CreateITensor(shape);
    Snpe_TensorShape_Delete(shape);
    if (!itensor) {
        Snpe_StringList_Delete(input_names);
        return write_error(fd, "failed to create ITensor");
    }

    const size_t expected = Snpe_ITensor_GetSize(itensor);
    if (input_floats != expected) {
        Snpe_ITensor_Delete(itensor);
        Snpe_StringList_Delete(input_names);
        return write_error(fd, "input size mismatch");
    }

    std::memcpy(Snpe_ITensor_GetData(itensor), input, expected * sizeof(float));

    Snpe_TensorMap_Handle_t input_map = Snpe_TensorMap_Create();
    Snpe_TensorMap_Add(input_map, input_name, itensor);
    Snpe_TensorMap_Handle_t output_map = Snpe_TensorMap_Create();

    const auto t0 = std::chrono::steady_clock::now();
    const Snpe_ErrorCode_t rc = Snpe_SNPE_ExecuteITensors(snpe, input_map, output_map);
    const auto t1 = std::chrono::steady_clock::now();
    const uint32_t infer_us = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    Snpe_ITensor_Delete(itensor);
    Snpe_TensorMap_Delete(input_map);
    Snpe_StringList_Delete(input_names);

    if (rc != SNPE_SUCCESS) {
        Snpe_TensorMap_Delete(output_map);
        return write_error(fd, Snpe_ErrorCode_GetLastErrorString());
    }

    Snpe_StringList_Handle_t out_names = Snpe_TensorMap_GetTensorNames(output_map);
    const uint16_t n_out = static_cast<uint16_t>(Snpe_StringList_Size(out_names));
    if (!write_ok_header(fd, infer_us, n_out)) {
        Snpe_StringList_Delete(out_names);
        Snpe_TensorMap_Delete(output_map);
        return false;
    }

    for (size_t i = 0; i < n_out; ++i) {
        const char* name = Snpe_StringList_At(out_names, i);
        const std::string name_str(name ? name : "output");
        Snpe_ITensor_Handle_t out_tensor = Snpe_TensorMap_GetTensor_Ref(output_map, name);
        const size_t n_floats = Snpe_ITensor_GetSize(out_tensor);
        const float* data = static_cast<const float*>(Snpe_ITensor_GetData(out_tensor));
        const uint32_t data_len = static_cast<uint32_t>(n_floats * sizeof(float));

        if (shm_outputs) {
            const std::string out_path = g_shm_dir + "/out_" + std::to_string(i) + ".bin";
            int out_fd = ::open(out_path.c_str(), O_RDWR | O_CREAT, 0600);
            if (out_fd < 0) {
                Snpe_StringList_Delete(out_names);
                Snpe_TensorMap_Delete(output_map);
                return write_error(fd, "failed to open shm output");
            }
            struct stat st {};
            if (::fstat(out_fd, &st) != 0 || static_cast<size_t>(st.st_size) < data_len) {
                if (::ftruncate(out_fd, static_cast<off_t>(data_len)) != 0) {
                    ::close(out_fd);
                    Snpe_StringList_Delete(out_names);
                    Snpe_TensorMap_Delete(output_map);
                    return write_error(fd, "failed to size shm output");
                }
            }
            if (::lseek(out_fd, 0, SEEK_SET) < 0 ||
                static_cast<size_t>(::write(out_fd, data, data_len)) != data_len) {
                ::close(out_fd);
                Snpe_StringList_Delete(out_names);
                Snpe_TensorMap_Delete(output_map);
                return write_error(fd, "failed to write shm output");
            }
            ::close(out_fd);
            if (!write_output_meta(fd, name_str, data_len)) {
                Snpe_StringList_Delete(out_names);
                Snpe_TensorMap_Delete(output_map);
                return false;
            }
        } else {
            const uint16_t name_len = htons(static_cast<uint16_t>(name_str.size()));
            const uint32_t len_be = htonl(data_len);
            if (!write_exact(fd, &name_len, 2) || !write_exact(fd, name_str.data(), name_str.size()) ||
                !write_exact(fd, &len_be, 4) || !write_exact(fd, data, data_len)) {
                Snpe_StringList_Delete(out_names);
                Snpe_TensorMap_Delete(output_map);
                return false;
            }
        }
    }

    Snpe_StringList_Delete(out_names);
    Snpe_TensorMap_Delete(output_map);
    return true;
}

bool handle_infer(int fd, Snpe_SNPE_Handle_t snpe, const std::vector<float>& input) {
    return execute_infer(snpe, input.data(), input.size(), fd, false);
}

bool handle_infer_shm(int fd, Snpe_SNPE_Handle_t snpe, uint32_t input_bytes) {
    if (g_shm_dir.empty()) {
        return write_error(fd, "daemon started without --shm-dir");
    }
    if (input_bytes == 0 || input_bytes % sizeof(float) != 0) {
        return write_error(fd, "invalid input length");
    }
    const std::string in_path = g_shm_dir + "/in.bin";
    const int in_fd = ::open(in_path.c_str(), O_RDONLY);
    if (in_fd < 0) {
        return write_error(fd, "failed to open shm input");
    }
    const size_t input_floats = input_bytes / sizeof(float);
    void* mapped =
        ::mmap(nullptr, input_bytes, PROT_READ, MAP_SHARED, in_fd, 0);
    ::close(in_fd);
    if (mapped == MAP_FAILED) {
        return write_error(fd, "mmap input failed");
    }
    const bool ok = execute_infer(
        snpe, static_cast<const float*>(mapped), input_floats, fd, true);
    ::munmap(mapped, input_bytes);
    return ok;
}

bool handle_client(int fd, Snpe_SNPE_Handle_t snpe) {
    while (true) {
        uint32_t magic = 0;
        uint8_t version = 0;
        uint8_t cmd = 0;
        if (!read_exact(fd, &magic, 4) || !read_exact(fd, &version, 1) || !read_exact(fd, &cmd, 1)) {
            return false;
        }
        if (magic != kMagic || version != kProtoVersion) {
            write_error(fd, "bad protocol header");
            return false;
        }

        if (cmd == kCmdShutdown) {
            return false;
        }
        if (cmd == kCmdPing) {
            if (!write_ok_header(fd, 0, 0)) return false;
            continue;
        }
        if (cmd == kCmdInferShm) {
            uint32_t input_len = 0;
            if (!read_exact(fd, &input_len, 4)) return false;
            if (!handle_infer_shm(fd, snpe, input_len)) return false;
            continue;
        }
        if (cmd != kCmdInfer) {
            write_error(fd, "unknown command");
            continue;
        }

        uint32_t input_len = 0;
        if (!read_exact(fd, &input_len, 4)) return false;
        if (input_len == 0 || input_len % sizeof(float) != 0) {
            write_error(fd, "invalid input length");
            continue;
        }

        std::vector<float> input(input_len / sizeof(float));
        if (!read_exact(fd, input.data(), input_len)) return false;
        if (!handle_infer(fd, snpe, input)) return false;
    }
}

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0
        << " --dlc PATH --socket PATH [--shm-dir PATH] [--runtime dsp]"
           " --graph NAME --outputs name1,name2 [--perf burst] [--init-cache|--no-init-cache]"
           " [--build-check] [--legacy-output-api] [--contract-version]\n";
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else if (c != ' ') {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dlc;
    std::string socket_path;
    std::string runtime_str = "dsp";
    std::string perf_str = "burst";
    std::string outputs_str;
    std::string graph;
    bool build_check = false;
    bool legacy_output_api = false;
    bool init_cache = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--dlc" || arg == "-d") && i + 1 < argc) {
            dlc = argv[++i];
        } else if ((arg == "--socket" || arg == "-s") && i + 1 < argc) {
            socket_path = argv[++i];
        } else if ((arg == "--runtime" || arg == "-r") && i + 1 < argc) {
            runtime_str = argv[++i];
        } else if ((arg == "--perf" || arg == "-p") && i + 1 < argc) {
            perf_str = argv[++i];
        } else if ((arg == "--shm-dir") && i + 1 < argc) {
            g_shm_dir = argv[++i];
        } else if ((arg == "--outputs" || arg == "-o") && i + 1 < argc) {
            outputs_str = argv[++i];
        } else if (arg == "--graph" && i + 1 < argc) {
            graph = argv[++i];
        } else if (arg == "--build-check") {
            build_check = true;
        } else if (arg == "--legacy-output-api") {
            legacy_output_api = true;
        } else if (arg == "--contract-version") {
            std::cout << "SNPE_DAEMON_GRAPH_DSP_V3" << std::endl;
            return 0;
        } else if (arg == "--init-cache" || arg == "-c") {
            init_cache = true;
        } else if (arg == "--no-init-cache") {
            init_cache = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "[snpe-daemon] unknown/incomplete option: " << arg << std::endl;
            return 1;
        }
    }

    if (runtime_str != "dsp" || graph.empty() || outputs_str.empty() ||
        (legacy_output_api && !build_check) || dlc.empty() || (!build_check && socket_path.empty())) {
        print_usage(argv[0]);
        return 1;
    }

    const std::vector<std::string> output_tensors = split_csv(outputs_str);

    Snpe_Util_InitializeLogging(SNPE_LOG_LEVEL_ERROR);

    Snpe_SNPE_Handle_t snpe = build_snpe(
        dlc,
        runtime_from_string(runtime_str),
        perf_from_string(perf_str),
        init_cache,
        output_tensors,
        graph,
        legacy_output_api);
    if (!snpe) return 2;
    Snpe_StringList_Handle_t actual = Snpe_SNPE_GetOutputTensorNames(snpe);
    bool output_contract = actual && Snpe_StringList_Size(actual) == output_tensors.size();
    for (const auto& expected : output_tensors) {
        bool found = false;
        for (size_t i = 0; actual && i < Snpe_StringList_Size(actual); ++i) {
            const char* name = Snpe_StringList_At(actual, i);
            if (name && expected == name) found = true;
        }
        output_contract = output_contract && found;
    }
    Snpe_StringList_Delete(actual);
    if (!output_contract) {
        std::cerr << "[snpe-daemon] built output names do not match boxes/scores contract" << std::endl;
        Snpe_SNPE_Delete(snpe);
        return 2;
    }
    if (build_check) {
        std::cout << "BUILD_CHECK_PASS selector=" << (legacy_output_api ? "legacy" : "network") << std::endl;
        Snpe_SNPE_Delete(snpe);
        return 0;
    }

    ::unlink(socket_path.c_str());
    const int srv = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) {
        perror("socket");
        Snpe_SNPE_Delete(snpe);
        return 3;
    }

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        std::cerr << "[snpe-daemon] socket path too long\n";
        Snpe_SNPE_Delete(snpe);
        return 3;
    }
    std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        ::close(srv);
        Snpe_SNPE_Delete(snpe);
        return 3;
    }
    if (::listen(srv, 4) < 0) {
        perror("listen");
        ::close(srv);
        Snpe_SNPE_Delete(snpe);
        return 3;
    }

    std::cerr << "[snpe-daemon] ready dlc=" << dlc << " socket=" << socket_path << std::endl;

    while (true) {
        const int client = ::accept(srv, nullptr, nullptr);
        if (client < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        handle_client(client, snpe);
        ::close(client);
    }

    ::close(srv);
    ::unlink(socket_path.c_str());
    Snpe_SNPE_Delete(snpe);
    return 0;
}
