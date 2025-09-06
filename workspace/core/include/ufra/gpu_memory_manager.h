#pragma once

#include "types.h"
#include <memory>
#include <string>

namespace ufra {

class GPUMemoryManager {
public:
    GPUMemoryManager();
    ~GPUMemoryManager();

    bool initialize(GPUBackend backend);
    void cleanup();
    
    size_t getAvailableMemory() const;
    size_t getTotalMemory() const;
    float getMemoryUtilization() const;
    
    void* allocateMemory(size_t bytes);
    void deallocateMemory(void* ptr);
    
    void enableMemoryPool(bool enable);
    void setMemoryPoolSize(size_t size_bytes);
    
    std::string getBackendInfo() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ufra