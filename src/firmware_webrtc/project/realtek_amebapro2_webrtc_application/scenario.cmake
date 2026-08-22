cmake_minimum_required(VERSION 3.6)

enable_language(C CXX ASM)

#MMF_MODULE
list(
    APPEND app_sources
    ${sdk_root}/component/media/mmfv2/module_video.c
    ${sdk_root}/component/media/mmfv2/module_rtsp2.c
    ${sdk_root}/component/media/mmfv2/module_array.c
    ${sdk_root}/component/media/mmfv2/module_audio.c
    ${sdk_root}/component/media/mmfv2/module_aac.c
    ${sdk_root}/component/media/mmfv2/module_aad.c
    ${sdk_root}/component/media/mmfv2/module_g711.c
    ${sdk_root}/component/media/mmfv2/module_httpfs.c
    ${sdk_root}/component/media/mmfv2/module_i2s.c
    ${sdk_root}/component/media/mmfv2/module_mp4.c
    ${sdk_root}/component/media/mmfv2/module_rtp.c
    ${sdk_root}/component/media/mmfv2/module_opusc.c
    ${sdk_root}/component/media/mmfv2/module_opusd.c
    ${sdk_root}/component/media/mmfv2/module_uvcd.c
    ${sdk_root}/component/media/mmfv2/module_demuxer.c
    ${sdk_root}/component/media/mmfv2/module_fmp4.c
    ${sdk_root}/component/media/mmfv2/module_fileloader.c
    ${sdk_root}/component/media/mmfv2/module_filesaver.c
    ${sdk_root}/component/media/mmfv2/module_queue.c
    ${sdk_root}/component/media/mmfv2/module_vipnn.c
    ${sdk_root}/component/video/osd2/osd_render.c
)

#NN MODEL + utils (face detection on NPU)
list(
    APPEND app_sources
    ${sdk_prj_example_root}/src/test_model/model_scrfd.c
    ${sdk_prj_example_root}/src/test_model/nn_utils/sigmoid.c
    ${sdk_prj_example_root}/src/test_model/nn_utils/quantize.c
    ${sdk_prj_example_root}/src/test_model/nn_utils/iou.c
    ${sdk_prj_example_root}/src/test_model/nn_utils/nms.c
    ${sdk_prj_example_root}/src/test_model/nn_utils/tensor.c
    ${sdk_prj_example_root}/src/test_model/nn_utils/class_name.c
)

#USER
list(
	APPEND scn_sources
	${prj_root}/src/main.c
)

if(UNITEST)
	include(${prj_root}/src/internal/unitest/unitest.cmake OPTIONAL)
endif()

list(
	APPEND scn_inc_path
	${app_example_inc_path}
	${prj_root}/src
	${sdk_prj_example_root}/src/${viplite}/sdk/inc
	${sdk_prj_example_root}/src/${viplite}/driver/inc
	${sdk_prj_example_root}/src/${viplite}/hal/inc
	${sdk_prj_example_root}/src/${viplite}/hal/user
	${sdk_prj_example_root}/src/${viplite}/hal/user/freeRTOS
	${sdk_prj_example_root}/src/test_model
)

list(
	APPEND scn_flags
	${app_example_flags}
)

list(
	APPEND scn_libs
	${app_example_lib}
)

list(
	APPEND scn_sources
	${app_example_sources}
)