function(generate_build_info)
  # Generate build information.
  string(TIMESTAMP BUILD_TIME "%Y-%m-%d %H:%M:%S")

  execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND git config --get remote.origin.url
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_ORIGIN
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND git status --porcelain
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(GIT_STATUS)
    set(GIT_DIRTY "-dirty")
  else()
    set(GIT_DIRTY "")
  endif()

  # Combine build time, commit hash, dirty flag, and origin.
  set(BUILD_INFO "${BUILD_TIME} (Commit: ${GIT_COMMIT_HASH}${GIT_DIRTY}, Origin: ${GIT_ORIGIN})")
  
  # Set the BUILD_INFO variable in the parent scope
  set(BUILD_INFO ${BUILD_INFO} PARENT_SCOPE)
  
  # Add the definition
  add_definitions(-DBUILD_INFO="${BUILD_INFO}")
endfunction()
