cmake_minimum_required(VERSION 3.16)
project(c_oloy C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(DET_ENABLE_ASAN "Enable AddressSanitizer" OFF)

add_library(kshira STATIC
  src/kshira_core.c
  src/kshira_quant.c
  src/kshira_sparse.c
  src/kshira_phase.c
  src/kshira_rad.c
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

add_library(det STATIC src/det.c src/det_io.c)
target_include_directories(det PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(det PUBLIC kshira)
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

add_executable(det_bench tools/bench.c)
if(MSVC)
  target_link_libraries(det_bench PRIVATE det)
else()
  target_link_libraries(det_bench PRIVATE det m)
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
  target_compile_options(det_bench PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(det_bench PRIVATE -fsanitize=address,undefined)
  target_compile_options(kshira_bench PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(kshira_bench PRIVATE -fsanitize=address,undefined)
  target_compile_options(kshira_domain_bench PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(kshira_domain_bench PRIVATE -fsanitize=address,undefined)
  target_compile_options(kshira_multiscale_bench PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(kshira_multiscale_bench PRIVATE -fsanitize=address,undefined)
endif()
