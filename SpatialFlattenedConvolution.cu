// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

__global__ void conv_1d_horizontal(const int n, float *y, const float *x, const float *w,
                                   const int iH, const int iW, const int kL)
{
  // horizontal convolution
  CUDA_KERNEL_LOOP(dst, n) {
    int oW = iW - kL + 1;
    int src_x = (dst/oW)*iW + dst%oW;
    int src_w = (dst/(oW*iH));

    for(int k = 0; k < kL; k++) {
      y[dst] += w[src_w + k]*x[src_x + k];
    }
  }
}

__global__ void conv_1d_vertical(const int n, float *y, const float *x, const float *w,
                                 const int iH, const int iW, const int kL)
{
  // vertical convolution
  CUDA_KERNEL_LOOP(dst, n) {
    int oH = iH - kL + 1;
    int src_x = (dst/(oH*iW))*iH*iW + dst%(oH*iW);
    int src_w = (dst/(oH*iW));

    for(int k = 0; k < kL; k++) {
      y[dst] += w[src_w + k]*x[src_x + k*iW];
    }
  }
}

static int cunn_SpatialFlattenedConvolution_updateOutput(lua_State *L) {
  // Input
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  // Params:
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THCudaTensor *weight_l = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight_l", "torch.CudaTensor");
  THCudaTensor *weight_v = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight_v", "torch.CudaTensor");
  THCudaTensor *weight_h = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight_h", "torch.CudaTensor");
  THCudaTensor *bias_l = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias_l", "torch.CudaTensor");
  THCudaTensor *bias_v = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias_v", "torch.CudaTensor");
  THCudaTensor *bias_h = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias_h", "torch.CudaTensor");

  THCudaTensor *intm1 = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "intm1", "torch.CudaTensor");
  THCudaTensor *intm2 = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "intm2", "torch.CudaTensor");

  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;


  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize4d(intm1, batchSize, nOutputPlane, inputHeight, inputWidth);
  THCudaTensor_resize4d(intm2, batchSize, nOutputPlane, outputHeight, outputWidth);
  THCudaTensor_resize4d(output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < inputHeight*inputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize2d(ones, inputHeight, inputWidth);
    THCudaTensor_fill(ones, 1);
  }

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new();
  THCudaTensor *intm1_n = THCudaTensor_new();
  THCudaTensor *intm2_n = THCudaTensor_new();
  THCudaTensor *output_n = THCudaTensor_new();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(input_n, input, 0, elt);
    THCudaTensor_select(intm1_n, intm1, 0, elt);
    THCudaTensor_select(intm2_n, intm2, 0, elt);
    THCudaTensor_select(output_n, output, 0, elt);


    // fill bias (column-major matrices)
    long m_l = nOutputPlane;
    long n_l = inputHeight * inputWidth;
    long k_l = 1;

    THCudaBlas_gemm('t', 'n', n_l, m_l, k_l, 1,
                    THCudaTensor_data(ones), k_l,
                    THCudaTensor_data(bias_l), k_l, 0,
                    THCudaTensor_data(intm1_n), n_l);

    long m_v = nOutputPlane;
    long n_v = outputHeight * inputWidth;
    long k_v = 1;

    THCudaBlas_gemm('t', 'n', n_v, m_v, k_v, 1,
                    THCudaTensor_data(ones), k_v,
                    THCudaTensor_data(bias_v), k_v, 0,
                    THCudaTensor_data(intm2_n), n_v);

    long m_h = nOutputPlane;
    long n_h = outputHeight * outputWidth;
    long k_h = 1;

    THCudaBlas_gemm('t', 'n', n_h, m_h, k_h, 1,
                    THCudaTensor_data(ones), k_h,
                    THCudaTensor_data(bias_h), k_h, 0,
                    THCudaTensor_data(output_n), n_h);


    // Lateral convolution
    long m = weight_l->size[0];
    long n = inputHeight*inputWidth;
    long k = weight_l->size[1];

    THCudaBlas_gemm('n', 'n', n, m, k, 1,
                    THCudaTensor_data(input_n), n,
                    THCudaTensor_data(weight_l), k, 1,
                    THCudaTensor_data(intm1_n), n);

    // Vertical convolution
    int num_kernels = nInputPlane * outputHeight * inputWidth;
    conv_1d_vertical <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>> (num_kernels,
                                                                        THCudaTensor_data(intm2_n),
                                                                        THCudaTensor_data(intm1_n),
                                                                        THCudaTensor_data(weight_v),
                                                                        inputHeight, inputWidth, kH);

    // Horizontal convolution
    num_kernels = nInputPlane * outputHeight * outputWidth;
    conv_1d_horizontal <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>> (num_kernels,
                                                                        THCudaTensor_data(output_n),
                                                                        THCudaTensor_data(intm2_n),
                                                                        THCudaTensor_data(weight_h),
                                                                        inputHeight, inputWidth, kW);
  }

  // Free
  THCudaTensor_free(input_n);
  THCudaTensor_free(intm1_n);
  THCudaTensor_free(intm2_n);
  THCudaTensor_free(output_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize3d(output, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(intm2, nInputPlane, outputHeight, inputWidth);
    THCudaTensor_resize3d(intm1, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
  }

  // return output
  return 1;
}

static const struct luaL_Reg cunn_SpatialFlattenedConvolution__ [] = {
  {"SpatialFlattenedConvolution_updateOutput", cunn_SpatialFlattenedConvolution_updateOutput},
  {NULL, NULL}
};

static void cunn_SpatialFlattenedConvolution_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialFlattenedConvolution__, "nn");
  lua_pop(L,1);
}
