#include "GPUFR/bb_gen.hpp"

#include "GPUFR/types.hpp"
#include "GPUFR/parser.hpp"
#include "GPUFR/nvrtc_helper.hpp"

#include <cstdlib>
#include <iostream>

#include <cuda.h>
#include <nvrtc.h>
#include <nvJitLink.h>

#define header_source HEADER_SRC
#define ff_math_obj FF_MATH_OBJ

void gen_bb_module(const std::string &expression, const std::vector<std::string> &vars, CUcontext cuda_context, CUdevice cuda_device, CUmodule &module){
	std::string cuda_code = cuda_from_expression(expression, vars);

	nvrtcProgram program;

	const char *header_name = "GPUFR/ff_math.cuh";
	NVRTC_SAFE_CALL(nvrtcCreateProgram(&program, cuda_code.data(), "black_box.cu", 0, NULL, NULL));

	std::string inc_dir_option = "--include-path=" header_source;
	const char *options[] = {inc_dir_option.c_str()};
	nvrtcResult compile_result = nvrtcCompileProgram(program, 1, options);

	if(compile_result != NVRTC_SUCCESS){
		size_t log_size;
		NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(program, &log_size));
		std::string log(log_size, '#');

		NVRTC_SAFE_CALL(nvrtcGetProgramLog(program, log.data()));

		std::cout << "Black Box compilation failed!\n"<<log<<"\n";

		exit(0);
	}

	size_t PTXSize;
	NVRTC_SAFE_CALL(nvrtcGetPTXSize(program, &PTXSize));
	char *PTX = new char[PTXSize];
	NVRTC_SAFE_CALL(nvrtcGetPTX(program, PTX));
	NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));
	
	nvJitLinkHandle handle;

	i32 major, minor;

	cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuda_device);
	cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuda_device);

	i32 arch = 10*major + minor;

	char smbuf[16];
	sprintf(smbuf, "-arch=sm_%d", arch);
	const char* link_options[] = {smbuf};

	NVJITLINK_SAFE_CALL(handle, nvJitLinkCreate(&handle, 1, link_options));

	std::cout<<ff_math_obj<<"\n";
	NVJITLINK_SAFE_CALL(handle, nvJitLinkAddFile(handle, NVJITLINK_INPUT_OBJECT, ff_math_obj)); 
	NVJITLINK_SAFE_CALL(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX, (void *)PTX, PTXSize, "bb"));

	NVJITLINK_SAFE_CALL(handle, nvJitLinkComplete(handle));

	size_t cubinSize;
	NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubinSize(handle, &cubinSize));
	void *cubin = malloc(cubinSize);
	NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubin(handle, cubin));
	NVJITLINK_SAFE_CALL(handle, nvJitLinkDestroy(&handle));

	cuModuleLoadData(&module, cubin);
}
