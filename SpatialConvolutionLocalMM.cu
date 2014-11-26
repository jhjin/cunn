// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

__global__ void zero_off_diagonal_gradients_kernel(const int n, float* gW, float* w_ind,
                                                   const int nInputPlane, const int klength, const int kC) {
  CUDA_KERNEL_LOOP(index, n) {
    int row = index / (klength * nInputPlane);
    int col = index % (klength * nInputPlane);

    int w_begin = (int)w_ind[row]*klength;
    int w_end = (int)w_ind[row]*klength + kC*klength - 1;

    if ((col < w_begin) | (col > w_end)) {
      gW[index] = 0.0f;
    }
  }
}

void zero_off_diagonal_gradients(float* gW, float* w_ind, const int nInputPlane, const int nOutputPlane,
                                 const int kC, const int kH, const int kW) {
  int num_kernels = nInputPlane * nOutputPlane * kH * kW;
  int klength = kH * kW;

  zero_off_diagonal_gradients_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>> (num_kernels, gW, w_ind, nInputPlane, klength, kC);
}

static int cunn_SpatialConvolutionLocalMM_updateOutput(lua_State *L) {
  // Input
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  // Params:
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int kC = luaT_getfieldcheckint(L, 1, "kC");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THCudaTensor *w_indicator = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "w_indicator", "torch.CudaTensor");
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *columns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
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
  THCudaTensor_resize4d(output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize2d(ones, outputHeight, outputWidth);
    THCudaTensor_fill(ones, 1);
  }

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new();
  THCudaTensor *output_n = THCudaTensor_new();

  // zero cross-layer weights
  zero_off_diagonal_gradients(THCudaTensor_data(weight), THCudaTensor_data(w_indicator),
                              nInputPlane, nOutputPlane, kC, kH, kW);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(input_n, input, 0, elt);
    THCudaTensor_select(output_n, output, 0, elt);

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
    long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
      't', 'n',
      n_, m_, k_,
      1,
      THCudaTensor_data(ones), k_,
      THCudaTensor_data(bias), k_,
      0,
      THCudaTensor_data(output_n), n_
    );

    // Extract columns:
    im2col(
           THCudaTensor_data(input_n),
           nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
           THCudaTensor_data(columns)
           );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[0];
    long n = columns->size[1];
    long k = weight->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
                'n', 'n',
                n, m, k,
                1,
                THCudaTensor_data(columns), n,
                THCudaTensor_data(weight), k,
                1,
                THCudaTensor_data(output_n), n
                );
  }

  // Free
  THCudaTensor_free(input_n);
  THCudaTensor_free(output_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize3d(output, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
  }

  // return output
  return 1;
}

static int cunn_SpatialConvolutionLocalMM_updateGradInput(lua_State *L) {
  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int kC = luaT_getfieldcheckint(L, 1, "kC");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");
  int cec = luaT_getfieldcheckboolean(L, 1, "cec");

  THCudaTensor *w_indicator = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "w_indicator", "torch.CudaTensor");
  THCudaTensor *weight = NULL;
  if (cec == 1) {
    weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "cecWeight", "torch.CudaTensor");
  } else {
    weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  }
  THCudaTensor *gradColumns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
    THCudaTensor_resize4d(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize4d(gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(gradColumns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new();
  THCudaTensor *gradInput_n = THCudaTensor_new();
  THCudaTensor *gradOutput_n = THCudaTensor_new();

  // zero cross-layer weights
  zero_off_diagonal_gradients(THCudaTensor_data(weight), THCudaTensor_data(w_indicator),
                              nInputPlane, nOutputPlane, kC, kH, kW);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THCudaTensor_select(input_n, input, 0, elt);
    THCudaTensor_select(gradInput_n, gradInput, 0, elt);
    THCudaTensor_select(gradOutput_n, gradOutput, 0, elt);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[1];
    long n = gradColumns->size[1];
    long k = weight->size[0];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
                'n', 't',
                n, m, k,
                1,
                THCudaTensor_data(gradOutput_n), n,
                THCudaTensor_data(weight), m,
                0,
                THCudaTensor_data(gradColumns), n
                );

    // Unpack columns back into input:
    col2im(
           THCudaTensor_data(gradColumns),
           nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
           THCudaTensor_data(gradInput_n)
           );
  }

  // Free
  THCudaTensor_free(input_n);
  THCudaTensor_free(gradInput_n);
  THCudaTensor_free(gradOutput_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize3d(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize3d(gradInput, nInputPlane, inputHeight, inputWidth);
  }

  // Return gradInput
  return 1;
}

static int cunn_SpatialConvolutionLocalMM_accGradParameters(lua_State *L) {
  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int kC = luaT_getfieldcheckint(L, 1, "kC");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");
  float scale = luaL_optnumber(L, 4, 1);

  THCudaTensor *w_indicator = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "w_indicator", "torch.CudaTensor");
  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
  THCudaTensor *columns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
    THCudaTensor_resize4d(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;



  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize2d(ones, outputHeight, outputWidth);
    THCudaTensor_fill(ones, 1);
  }

  // Resize temporary columns
  THCudaTensor_resize2d(columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new();
  THCudaTensor *gradOutput_n = THCudaTensor_new();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(input_n, input, 0, elt);
    THCudaTensor_select(gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    im2col(
           THCudaTensor_data(input_n),
           nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
           THCudaTensor_data(columns)
           );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = gradWeight->size[0];
    long n = gradWeight->size[1];
    long k = columns->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
                't', 'n',
                n, m, k,
                scale,
                THCudaTensor_data(columns), k,
                THCudaTensor_data(gradOutput_n), k,
                1,
                THCudaTensor_data(gradWeight), n
                );

    // zero cross-layer gradWeights
    zero_off_diagonal_gradients(THCudaTensor_data(gradWeight), THCudaTensor_data(w_indicator),
                                nInputPlane, nOutputPlane, kC, kH, kW);

    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = 1;
    long k_ = outputHeight * outputWidth;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
      'n', 'n',
      n_, m_, k_,
      scale,
      THCudaTensor_data(ones), n_,
      THCudaTensor_data(gradOutput_n), k_,
      1,
      THCudaTensor_data(gradBias), n_
    );
  }

  // Free
  THCudaTensor_free(input_n);
  THCudaTensor_free(gradOutput_n);

  // Resize
  if (batch == 0) {
    THCudaTensor_resize3d(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
  }

  // Return nothing
  return 0;
}

static const struct luaL_Reg cunn_SpatialConvolutionLocalMM__ [] = {
  {"SpatialConvolutionLocalMM_updateOutput", cunn_SpatialConvolutionLocalMM_updateOutput},
  {"SpatialConvolutionLocalMM_updateGradInput", cunn_SpatialConvolutionLocalMM_updateGradInput},
  {"SpatialConvolutionLocalMM_accGradParameters", cunn_SpatialConvolutionLocalMM_accGradParameters},
  {NULL, NULL}
};

static void cunn_SpatialConvolutionLocalMM_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialConvolutionLocalMM__, "nn");
  lua_pop(L,1);
}
