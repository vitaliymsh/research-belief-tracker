#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# PyCUDA CUDA Kernel for direct GPU cropping, resizing, normalization and channel transposition (HWC BGR -> NCHW RGB)
cuda_crop_src = """
__global__ void crop_and_normalize_kernel(
    const unsigned char* input_frame,  // 720x1280x3 (HWC, BGR)
    const float* bboxes,                // 4 x 4 (x1, y1, w, h)
    float* output_tensor,               // 4 x 3 x 256 x 128 (NCHW, RGB)
    int img_w, int img_h
) {
    int w_out = threadIdx.x + blockIdx.x * blockDim.x; // 0..127
    int h_out = threadIdx.y + blockIdx.y * blockDim.y; // 0..255
    int n_out = blockIdx.z;                            // 0..3

    if (w_out >= 128 || h_out >= 256 || n_out >= 4) return;

    // Get bounding box for this entity: [x1, y1, w, h]
    float x1 = bboxes[n_out * 4 + 0];
    float y1 = bboxes[n_out * 4 + 1];
    float w  = bboxes[n_out * 4 + 2];
    float h  = bboxes[n_out * 4 + 3];

    // Map output coordinates (h_out, w_out) to input frame coordinates
    float src_x = x1 + (w_out / 127.0f) * w;
    float src_y = y1 + (h_out / 255.0f) * h;

    int ix = (int)roundf(src_x);
    int iy = (int)roundf(src_y);

    // Clamp coordinates to input frame boundaries
    ix = max(0, min(ix, img_w - 1));
    iy = max(0, min(iy, img_h - 1));

    // Get pixel color from BGR input frame
    int in_idx = (iy * img_w + ix) * 3;
    float b = (float)input_frame[in_idx + 0] / 255.0f;
    float g = (float)input_frame[in_idx + 1] / 255.0f;
    float r = (float)input_frame[in_idx + 2] / 255.0f;

    // Normalize (RGB mean/std)
    float r_norm = (r - 0.485f) / 0.229f;
    float g_norm = (g - 0.456f) / 0.224f;
    float b_norm = (b - 0.406f) / 0.225f;

    // Write to output tensor in NCHW format
    int channel_stride = 256 * 128;
    int batch_stride = 3 * channel_stride;
    int pixel_idx = h_out * 128 + w_out;

    output_tensor[n_out * batch_stride + 0 * channel_stride + pixel_idx] = r_norm;
    output_tensor[n_out * batch_stride + 1 * channel_stride + pixel_idx] = g_norm;
    output_tensor[n_out * batch_stride + 2 * channel_stride + pixel_idx] = b_norm;
}
"""

class TRTEngine:
    def __init__(self, engine_path, cuda_context):
        self.engine_path = engine_path
        self.ctx = cuda_context
        
        self.ctx.push()
        try:
            self.logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            
            self.inputs = []
            self.outputs = []
            self.allocations = []
            
            for i in range(self.engine.num_bindings):
                is_input = self.engine.binding_is_input(i)
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = self.engine.get_binding_shape(i)
                size = int(np.prod(shape))
                
                # Allocate GPU memory
                allocation = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
                
                binding = {
                    'index': i,
                    'name': name,
                    'dtype': dtype,
                    'shape': shape,
                    'allocation': allocation,
                    'size': size
                }
                
                self.allocations.append(int(allocation))
                if is_input:
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)
        finally:
            self.ctx.pop()

    def run_async(self, input_buffers, output_buffers, stream):
        bindings = [None] * self.engine.num_bindings
        for binding in self.inputs:
            bindings[binding['index']] = int(input_buffers[binding['index']])
        for binding in self.outputs:
            bindings[binding['index']] = int(output_buffers[binding['index'] - len(self.inputs)])
            
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

def main():
    print("====================================================")
    print("Zero-Copy GPU-Resident Crop + batched OSNet Pipeline")
    print("====================================================")
    
    # Ensure nvcc is in path for PyCUDA compilation on Jetson
    os.environ['PATH'] = os.environ.get('PATH', '') + ':/usr/local/cuda/bin'
    
    yolo_path = "models/yolo26n.engine"
    osnet_path = "models/osnet_x0_25_b4_opt_fp16.engine"
    
    if not os.path.exists(yolo_path) or not os.path.exists(osnet_path):
        print("Missing engine files in models directory.")
        sys.exit(1)
        
    print("Initializing PyCUDA and compiling crop kernel...")
    cuda.init()
    device = cuda.Device(0)
    ctx = device.retain_primary_context()
    
    ctx.push()
    try:
        # Compile the CUDA custom crop-and-normalize kernel
        mod = SourceModule(cuda_crop_src)
        crop_kernel = mod.get_function("crop_and_normalize_kernel")
        print("CUDA Kernel compiled successfully.")
        
        print("Loading TensorRT Engines...")
        yolo_engine = TRTEngine(yolo_path, ctx)
        osnet_engine = TRTEngine(osnet_path, ctx)
        print("Engines loaded successfully.")
        
        # Shapes
        yolo_in_shape = yolo_engine.inputs[0]['shape']
        yolo_out_shape = yolo_engine.outputs[0]['shape']
        osnet_out_shape = osnet_engine.outputs[0]['shape']
        
        # GPU Input Frame: 720x1280x3 uint8 (BGR format)
        img_h, img_w, img_c = 720, 1280, 3
        frame_cpu = (np.random.rand(img_h, img_w, img_c) * 255).astype(np.uint8)
        frame_gpu = cuda.mem_alloc(frame_cpu.nbytes)
        cuda.memcpy_htod(frame_gpu, frame_cpu)
        
        # Bounding boxes representing YOLO results: 4 crops [x, y, w, h]
        bboxes_cpu = np.array([
            [100, 150, 120, 200],
            [300, 100, 150, 250],
            [600, 200, 110, 180],
            [800, 350, 130, 220]
        ], dtype=np.float32)
        bboxes_gpu = cuda.mem_alloc(bboxes_cpu.nbytes)
        
        # Outputs
        yolo_output_host = np.empty(yolo_out_shape, dtype=np.float32)
        osnet_output_host = np.empty(osnet_out_shape, dtype=np.float32)
        
        stream = cuda.Stream()
        
        # Warmup runs
        print("Warming up entire GPU pipeline...")
        for _ in range(10):
            # 1. Run YOLO (Upload resized 640x640 mock input)
            yolo_mock_input = np.random.rand(*yolo_in_shape).astype(np.float32)
            cuda.memcpy_htod_async(yolo_engine.inputs[0]['allocation'], yolo_mock_input, stream)
            yolo_engine.run_async([yolo_engine.inputs[0]['allocation']], [yolo_engine.outputs[0]['allocation']], stream)
            
            # 2. Upload Bounding Boxes (simulate CPU-to-GPU tiny transfer after YOLO NMS)
            cuda.memcpy_htod_async(bboxes_gpu, bboxes_cpu, stream)
            
            # 3. GPU Crop and Normalize Kernel
            block_dim = (16, 16, 1)
            grid_dim = (8, 16, 4) # 128/16 = 8, 256/16 = 16, Batch size = 4
            crop_kernel(
                frame_gpu, bboxes_gpu, osnet_engine.inputs[0]['allocation'],
                np.int32(img_w), np.int32(img_h),
                block=block_dim, grid=grid_dim, stream=stream
            )
            
            # 4. Run OSNet ReID
            osnet_engine.run_async([osnet_engine.inputs[0]['allocation']], [osnet_engine.outputs[0]['allocation']], stream)
            
            # Download outputs to synchronize
            cuda.memcpy_dtoh_async(yolo_output_host, yolo_engine.outputs[0]['allocation'], stream)
            cuda.memcpy_dtoh_async(osnet_output_host, osnet_engine.outputs[0]['allocation'], stream)
            stream.synchronize()
            
        print("Warmup complete. Starting benchmark...")
        
        # Benchmark loop
        num_frames = 100
        start_time = time.perf_counter()
        
        yolo_latencies = []
        crop_latencies = []
        osnet_latencies = []
        frame_latencies = []
        
        # Generate input once outside the loop to avoid CPU generation overhead
        yolo_mock_input = np.random.rand(*yolo_in_shape).astype(np.float32)
        
        for frame_idx in range(num_frames):
            frame_start = time.perf_counter()
            
            # 1. Run YOLO (simulates inference on current frame)
            t_yolo_start = time.perf_counter()
            cuda.memcpy_htod_async(yolo_engine.inputs[0]['allocation'], yolo_mock_input, stream)
            yolo_engine.run_async([yolo_engine.inputs[0]['allocation']], [yolo_engine.outputs[0]['allocation']], stream)
            stream.synchronize()
            t_yolo_end = time.perf_counter()
            yolo_latencies.append((t_yolo_end - t_yolo_start) * 1000.0)
            
            # 2. Crop, Resize, Normalize & Transpose on GPU using custom CUDA kernel
            t_crop_start = time.perf_counter()
            cuda.memcpy_htod_async(bboxes_gpu, bboxes_cpu, stream)
            block_dim = (16, 16, 1)
            grid_dim = (8, 16, 4)
            crop_kernel(
                frame_gpu, bboxes_gpu, osnet_engine.inputs[0]['allocation'],
                np.int32(img_w), np.int32(img_h),
                block=block_dim, grid=grid_dim, stream=stream
            )
            stream.synchronize()
            t_crop_end = time.perf_counter()
            crop_latencies.append((t_crop_end - t_crop_start) * 1000.0)
            
            # 3. Run Batched OSNet on GPU on the newly cropped data
            t_osnet_start = time.perf_counter()
            osnet_engine.run_async([osnet_engine.inputs[0]['allocation']], [osnet_engine.outputs[0]['allocation']], stream)
            stream.synchronize()
            t_osnet_end = time.perf_counter()
            osnet_latencies.append((t_osnet_end - t_osnet_start) * 1000.0)
            
            frame_end = time.perf_counter()
            frame_latencies.append((frame_end - frame_start) * 1000.0)
            
        total_duration = time.perf_counter() - start_time
        
        # Metrics
        avg_yolo = np.mean(yolo_latencies)
        avg_crop = np.mean(crop_latencies)
        avg_osnet = np.mean(osnet_latencies)
        avg_frame = np.mean(frame_latencies)
        fps = num_frames / total_duration
        
        print("\n================== PIPELINE BENCHMARK RESULTS ==================")
        print(f"Frames processed: {num_frames}")
        print(f"ReID Batch Size (Crops): 4")
        print("-----------------------------------------------------------------")
        print(f"Average YOLO Inference:       {avg_yolo:.2f} ms")
        print(f"Average GPU Crop + Normalize: {avg_crop:.2f} ms  <-- [ZERO COPY SPEED]")
        print(f"Average OSNet (batch of 4):   {avg_osnet:.2f} ms")
        print("-----------------------------------------------------------------")
        print(f"Average Processing Time per Frame: {avg_frame:.2f} ms")
        print(f"Combined Pipeline Throughput:       {fps:.2f} FPS")
        print("=================================================================")
        
        # Free allocated GPU arrays
        frame_gpu.free()
        bboxes_gpu.free()
        
    finally:
        ctx.pop()

if __name__ == "__main__":
    main()
