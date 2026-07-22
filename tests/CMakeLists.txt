# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Make Google Test available
FetchContent_MakeAvailable(googletest)

# Create test executable
add_executable(UnitTests
    tests.cxx
)

# Link against Google Test and the core library
target_link_libraries(UnitTests
    PRIVATE
        gtest_main
        gmock
        waveguide_core
)

# Set output directory for test executable
set_target_properties(UnitTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Discover tests for CTest
include(GoogleTest)
gtest_discover_tests(UnitTests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Add a custom test target for convenience
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS UnitTests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running unit tests..."
)
