set(IMGUI_ROOT ${LIB_DIR}/imgui)

set(IMGUI_SRC
    ${IMGUI_ROOT}/imgui.cpp
    ${IMGUI_ROOT}/imgui.h
    ${IMGUI_ROOT}/imgui_demo.cpp
    ${IMGUI_ROOT}/imgui_draw.cpp
    ${IMGUI_ROOT}/imgui_internal.h
    ${IMGUI_ROOT}/imgui_tables.cpp
    ${IMGUI_ROOT}/imgui_widgets.cpp
    ${IMGUI_ROOT}/imstb_rectpack.h
    ${IMGUI_ROOT}/imstb_textedit.h
    ${IMGUI_ROOT}/imstb_truetype.h
    ${IMGUI_ROOT}/backends/imgui_impl_glfw.h
    ${IMGUI_ROOT}/backends/imgui_impl_glfw.cpp
    ${IMGUI_ROOT}/backends/imgui_impl_opengl3.cpp
    ${IMGUI_ROOT}/backends/imgui_impl_opengl3.h
    )

add_library(libimgui)
target_sources(libimgui PUBLIC ${IMGUI_SRC})

target_include_directories(libimgui PUBLIC ${LIB_DIR}/imgui)
target_include_directories(libimgui PUBLIC ${LIB_DIR}/imgui/backends)

