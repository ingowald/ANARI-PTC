# build interactive viewer IF (and only if) there is a clone of the
# owl repo checked out as a sister directory to this project
if (EXISTS ../../owl)
  add_subdirectory(../../owl build_owl EXCLUDE_FROM_ALL)
  add_executable(testDeepViewer
    testDeepViewer.cpp
    testRenderer.cu
    )
  target_link_libraries(testDeepViewer
    deepCompositing
    owl_viewer
    stdc++fs # for experimental::filesystem
    )
  target_include_directories(testDeepViewer
    PUBLIC
    3rdParty/
    )
endif()




if (NOT (TARGET stb_image))
  add_subdirectory(3rdParty/stb_image EXCLUDE_FROM_ALL)
endif()

add_executable(testDeepOffline
  testDeepOffline.cpp
  testRenderer.cu
  )
target_link_libraries(testDeepOffline
  deepCompositing
  stb_image
  )
target_include_directories(testDeepOffline
  PUBLIC
  3rdParty/
  )





add_executable(dcGenTestData
  genTestData.cpp
  )
target_link_libraries(dcGenTestData
  deepCompositing
  stb_image
  )
target_include_directories(dcGenTestData
  PUBLIC
  3rdParty/
  )


