import torch
import gc

def check_memory():
    print("Allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("Reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("Total: %fGB"%(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024))


def clear_all_cuda_memory():
    # Ensure all CUDA operations are complete
    torch.cuda.synchronize()
    
    # Empty the cache on all devices
    for device_id in range(torch.cuda.device_count()):
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    
    # Clear references to any tensors and force garbage collection
    gc.collect()
    
    # Optionally, reset the CUDA context (commented out as it's more drastic and may not always be necessary)
    # for device_id in range(torch.cuda.device_count()):
    #     torch.cuda.reset()
        
    print("All CUDA memory cleared on all devices.")