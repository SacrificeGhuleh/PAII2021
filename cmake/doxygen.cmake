find_package(Doxygen)

if (DOXYGEN_FOUND)
  set(DOXYGEN_INPUT ${SOURCE_FILES})
  set(DOXYGEN_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
  
  add_custom_command(
    OUTPUT ${DOXYGEN_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
    COMMAND ${CMAKE_COMMAND} -E echo "Done."
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${DOXYGEN_INPUT}
  )
  
  add_custom_target(apidoc ALL DEPENDS ${DOXYGEN_OUTPUT})
  
  add_custom_target(apidoc_forced
                    COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
                    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
                    COMMAND ${CMAKE_COMMAND} -E echo "Done."
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    )
endif (DOXYGEN_FOUND)