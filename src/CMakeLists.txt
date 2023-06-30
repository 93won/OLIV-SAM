file(GLOB LIO "LIO/*.cpp")

file(GLOB VIO "VIO/GraphSolver.cpp", "VIO/TrackerOmni.cpp")

add_library(slvi_sam SHARED
Config.cpp
Viewer.cpp
IOUtils.cpp
${LIO}
${VIO}
)

target_link_libraries(slvi_sam ${THIRD_PARTY_LIBS})
