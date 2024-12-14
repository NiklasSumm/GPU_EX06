/**************************************************************************************************
 *
 *       Computer Engineering Group, Heidelberg University - GPU Computing Exercise 06
 *
 *                 Gruppe : TODO
 *
 *                   File : kernel.cu
 *
 *                Purpose : Reduction
 *
 **************************************************************************************************/

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

//
// Reduction_Kernel
//
__global__ void
reduction_Kernel(int numElements, float* dataIn, float* dataOut)
{
	extern __shared__ int sPartials[];

	float blockSum = 0.0;

	int elemRoundUp = numElements;

	if (elemRoundUp % blockDim.x != 0){
		elemRoundUp += blockDim.x - elemRoundUp % blockDim.x;
	}

	const int tid = threadIdx.x;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elemRoundUp; i += blockDim.x){
		if (i < numElements){
			sPartials[tid] = dataIn[i];
		}
		else{
			sPartials[tid] = 0;
		}

		__syncthreads();

		for ( unsigned int s = 1; s < blockDim.x; s *= 2 ) {
			if ( tid % ( 2 * s ) == 0 ) {
				sPartials[tid] += sPartials[tid + s];
			}

		__syncthreads();
		}

		blockSum += sPartials[0];
	}

	if ( tid == 0 ) {
		dataOut[blockIdx.x] = blockSum;
	}


  	//extern __shared__ float sh_Data[];
//
	//int iterations = numElementsShared + blockDim.x + 1 / blockDim.x;
//
	//for (int i = 0; i < iterations; i++){
	//	int elementId = iterations * blockIdx.x * blockDim.x + threadIdx.x + i * blockDim.x;
	//	int sharedId = threadIdx.x + i * blockDim.x;
	//	if (sharedId < numElementsShared){
	//		if (elementId < numElements){
	//			sh_Data[sharedId] = dataIn[elementId];
	//		}
	//		else{
	//			sh_Data[sharedId] = 0;
	//		}
	//	}
	//}
//
  	//__syncthreads();
//
  	//for (unsigned int s = 1; s < blockDim.x * iterations; s *= 2) {
	//	for (int i = 0; i < iterations; i++){
	//		int id = threadIdx.x + i * blockDim.x;
    //		if ((id % (2 * s)) == 0) {
	//			if (id + s < numElementsShared){
	//				sh_Data[id] += sh_Data[id + s];
	//			}
    //		}
  	//	}
	//	__syncthreads();
	//}
//
  	//if (threadIdx.x == 0) dataOut[blockIdx.x] = sh_Data[0];
}

void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut) {
	int sharedMemSize = blockSize.x * sizeof(float);
	reduction_Kernel<<< gridSize, blockSize, sharedMemSize>>>(numElements, dataIn, dataOut);
}

__global__ void
reduction_Kernel_improved(int numElements, float* dataIn, float* dataOut)
{
	extern __shared__ int sPartials[];

	float blockSum = 0.0;

	int elemRoundUp = numElements;

	if (elemRoundUp % blockDim.x != 0){
		elemRoundUp += blockDim.x - elemRoundUp % blockDim.x;
	}

	const int tid = threadIdx.x;

  	for (int i = 2 * blockIdx.x * blockDim.x + threadIdx.x; i < elemRoundUp; i += 2 * blockDim.x){
		if (i < numElements){
			if (i + blockDim.x < numElements){
				sPartials[tid] = dataIn[i] + dataIn[i + blockDim.x];
			}
			else{
				sPartials[tid] = dataIn[i];
			}
		}
		else{
			sPartials[tid] = 0;
		}

  		__syncthreads();

  		for ( unsigned int o = blockDim.x / 2; o > 0; o >>= 1 ) {
			if (tid < o ) {
				sPartials[tid] += sPartials[tid + o];
			}
			__syncthreads();
		}

		blockSum += sPartials[0]
	}

  	if (tid == 0) dataOut[blockIdx.x] = blockSum;
}

void reduction_Kernel_improved_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut) {
	int sharedMemSize = numElements * sizeof(float) / (gridSize.x * 2);
	reduction_Kernel_improved<<< gridSize, blockSize, sharedMemSize>>>(numElements, dataIn, dataOut);
}

//
// Reduction Kernel using CUDA Thrust
//

void thrust_reduction_Wrapper(int numElements, float* dataIn, float* dataOut) {
	thrust::device_ptr<float> in_ptr = thrust::device_pointer_cast(dataIn);
	thrust::device_ptr<float> out_ptr = thrust::device_pointer_cast(dataOut);
	
	*out_ptr = thrust::reduce(in_ptr, in_ptr + numElements, (float) 0., thrust::plus<float>());	
}
