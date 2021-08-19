cmake_minimum_required(VERSION 3.0)

PROJECT(MTCNN-PureC)

INCLUDE_DIRECTORIES(./include)

AUX_SOURCE_DIRECTORY(src DIR_SRCS)

ADD_EXECUTABLE(${PROJECT_NAME} ${DIR_SRCS})

if(CMAKE_HOST_UNIX)
    target_link_libraries(${PROJECT_NAME} m)
endif()

file(GLOB MODEL "${PROJECT_SOURCE_DIR}/model/*.txt")
add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   ${MODEL}
                   $<TARGET_FILE_DIR:${PROJECT_NAME}>)
                   
add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   "${PROJECT_SOURCE_DIR}/input1.txt"
                   $<TARGET_FILE_DIR:${PROJECT_NAME}>)