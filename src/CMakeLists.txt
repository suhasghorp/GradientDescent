
set(SOURCE_FILES
        GradientDescent.h
        GradientDescent.cpp
        )

add_library(GradientDescent SHARED STATIC ${SOURCE_FILES})
target_include_directories(GradientDescent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS GradientDescent DESTINATION ${GD_INSTALL_LIB_DIR})
install(FILES GradientDescent.h DESTINATION ${GD_INSTALL_INCLUDE_DIR})