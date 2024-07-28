#ifndef __DEVICE_OBJECT_H__
#define __DEVICE_OBJECT_H__

#include "driver_types.h"
#include <cuda_runtime_api.h>

template<typename Self> class DeviceObject
{
  public:
    __host__ [[nodiscard]] Self *new_device_ptr() const
    {
        Self *ptr = nullptr;
        cudaMalloc(&ptr, sizeof(Self));
        cudaMemcpy(ptr, static_cast<const Self *>(this), sizeof(Self), cudaMemcpyHostToDevice);
        return ptr;
    }

    __host__ void update_device_ptr(Self *ptr) const
    {
        cudaMemcpy(ptr, static_cast<const Self *>(this), sizeof(Self), cudaMemcpyHostToDevice);
    }
};

#endif
