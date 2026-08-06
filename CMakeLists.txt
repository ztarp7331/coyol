cmake_minimum_required(VERSION 3.16)
project(c_oloy C)

# The detector is a performance-sensitive CPU runtime.  Keep debug builds
# explicit; a plain configure should measure the optimized kernels.
if(NOT CMAKE_BUILD_TYPE AND NOT MSVC)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(DET_ENABLE_ASAN "Enable AddressSanitizer" OFF)

add_library(kshira STATIC
  src/kshira_core.c
  src/kshira_quant.c
  src/kshira_sparse.c
  src/kshira_solver.c
  src/kshira_budget.c
  src/kshira_phase.c
    src/kshira_rad.c
    src/kshira_pyramid.c
    src/kshira_rad_state.c
  src/kshira_session.c
  src/kshira_domain.c
  src/kshira_eval.c
)
target_include_directories(kshira PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
  target_compile_options(kshira PRIVATE /W4)
else()
  target_compile_options(kshira PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()
if(DET_ENABLE_ASAN AND NOT MSVC)
  target_compile_options(kshira PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(kshira PRIVATE -fsanitize=address,undefined)
endif()

add_library(native_graph STATIC
  src/native_graph_ops.c
  src/native_graph_model.c
  src/native_graph_model_f32.c
  src/native_graph_loader.c
  src/native_graph_post.c
)
target_include_directories(native_graph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(native_graph PUBLIC c_std_99)
if(MSVC)
  target_compile_options(native_graph PRIVATE /W4)
else()
  target_compile_options(native_graph PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

add_library(native_multiscale STATIC src/native_multiscale.c src/native_multiscale_f32.c)
target_include_directories(native_multiscale PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(native_multiscale PUBLIC native_graph)
target_compile_features(native_multiscale PUBLIC c_std_99)
if(MSVC)
  target_compile_options(native_multiscale PRIVATE /W4)
else()
  target_compile_options(native_multiscale PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

add_library(det STATIC src/det.c src/det_io.c)
target_include_directories(det PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(det PUBLIC kshira native_graph native_multiscale)
if(MSVC)
  target_compile_options(det PRIVATE /W4)
else()
  target_compile_options(det PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()
if(DET_ENABLE_ASAN AND NOT MSVC)
  target_compile_options(det PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(det PRIVATE -fsanitize=address,undefined)
endif()

enable_testing()
add_executable(det_tests tests/test_det.c)
if(MSVC)
  target_link_libraries(det_tests PRIVATE det kshira)
else()
  target_link_libraries(det_tests PRIVATE det kshira m)
endif()
add_test(NAME det_tests COMMAND det_tests)

add_executable(kshira_tests tests/test_kshira.c)
if(MSVC)
  target_link_libraries(kshira_tests PRIVATE kshira)
else()
  target_link_libraries(kshira_tests PRIVATE kshira m)
endif()
add_test(NAME kshira_tests COMMAND kshira_tests)

add_executable(kshira_learning_tests tests/test_kshira_learning.c)
if(MSVC)
  target_link_libraries(kshira_learning_tests PRIVATE det kshira)
else()
  target_link_libraries(kshira_learning_tests PRIVATE det kshira m)
endif()
add_test(NAME kshira_learning_tests COMMAND kshira_learning_tests)

if(MSVC)
  target_compile_options(det_tests PRIVATE /U NDEBUG)
  target_compile_options(kshira_tests PRIVATE /U NDEBUG)
  target_compile_options(kshira_learning_tests PRIVATE /U NDEBUG)
else()
  target_compile_options(det_tests PRIVATE -UNDEBUG)
  target_compile_options(kshira_tests PRIVATE -UNDEBUG)
  target_compile_options(kshira_learning_tests PRIVATE -UNDEBUG)
endif()

add_executable(det_bench tools/bench.c)
if(MSVC)
  target_link_libraries(det_bench PRIVATE det)
else()
  target_link_libraries(det_bench PRIVATE det m)
endif()

add_executable(native_graph_detect tools/native_graph_detect.c)
if(MSVC)
  target_link_libraries(native_graph_detect PRIVATE native_graph)
else()
  target_link_libraries(native_graph_detect PRIVATE native_graph m)
endif()

add_executable(native_multiscale_detect tools/native_multiscale_detect.c)
if(MSVC)
  target_link_libraries(native_multiscale_detect PRIVATE native_multiscale native_graph)
else()
  target_link_libraries(native_multiscale_detect PRIVATE native_multiscale native_graph m)
endif()

add_executable(native_lifecycle_smoke tools/native_lifecycle_smoke.c)
if(MSVC)
  target_link_libraries(native_lifecycle_smoke PRIVATE det)
else()
  target_link_libraries(native_lifecycle_smoke PRIVATE det m)
endif()

add_executable(native_adapter_bench tools/native_adapter_bench.c)
if(MSVC)
  target_link_libraries(native_adapter_bench PRIVATE det)
else()
  target_link_libraries(native_adapter_bench PRIVATE det m)
endif()

add_executable(native_teacher_manifest tools/native_teacher_manifest.c)
if(MSVC)
  target_link_libraries(native_teacher_manifest PRIVATE det)
else()
  target_link_libraries(native_teacher_manifest PRIVATE det m)
endif()

add_executable(viz_detect tools/viz_detect.c)
if(MSVC)
  target_link_libraries(viz_detect PRIVATE det)
else()
  target_link_libraries(viz_detect PRIVATE det m)
endif()

add_executable(kshira_bench tools/kshira_bench.c)
if(MSVC)
  target_link_libraries(kshira_bench PRIVATE kshira)
else()
  target_link_libraries(kshira_bench PRIVATE kshira m)
endif()

add_executable(kshira_domain_bench tools/kshira_domain_bench.c)
if(MSVC)
  target_link_libraries(kshira_domain_bench PRIVATE kshira)
else()
  target_link_libraries(kshira_domain_bench PRIVATE kshira m)
endif()

add_executable(kshira_multiscale_bench tools/kshira_multiscale_bench.c)
if(MSVC)
  target_link_libraries(kshira_multiscale_bench PRIVATE kshira)
else()
  target_link_libraries(kshira_multiscale_bench PRIVATE kshira m)
endif()

if(DET_ENABLE_ASAN AND NOT MSVC)
  target_compile_options(det_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(det_tests PRIVATE -fsanitize=address,undefined)
  target_compile_options(kshira_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(kshira_tests PRIVATE -fsanitize=address,undefined)
  target_compile_options(kshira_learning_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(kshira_learning_tests PRIVATE -fsanitize=address,undefined)
  target_compile_options(det_bench PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(det_bench PRIVATE -fsanitize=address,undefined)
  target_compile_options(kshira_bench PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(kshira_bench PRIVATE -fsanitize=address,undefined)
  target_compile_options(kshira_domain_bench PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(kshira_domain_bench PRIVATE -fsanitize=address,undefined)
  target_compile_options(kshira_multiscale_bench PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(kshira_multiscale_bench PRIVATE -fsanitize=address,undefined)
endif()
