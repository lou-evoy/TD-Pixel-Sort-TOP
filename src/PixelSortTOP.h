// Pixel Sort TOP — TD glue
// validated: TouchDesigner 2025.32050, TOP C++ API v12
#ifndef PIXELSORT_TOP_H
#define PIXELSORT_TOP_H

#include "TOP_CPlusPlusBase.h"
#include "cuda_runtime.h"
#include "PixelSortCUDA.h"

using namespace TD;

class PixelSortTOP : public TOP_CPlusPlusBase
{
public:
    PixelSortTOP(const OP_NodeInfo* info, TOP_Context* context);
    virtual ~PixelSortTOP();

    virtual void    getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void* reserved1) override;
    virtual void    execute(TOP_Output*, const OP_Inputs*, void* reserved1) override;

    virtual void    getErrorString(OP_String* error, void* reserved1) override;
    virtual void    getInfoPopupString(OP_String* info, void* reserved1) override;

    virtual void    setupParameters(OP_ParameterManager* manager, void* reserved1) override;

private:
    const OP_NodeInfo*  myNodeInfo;
    TOP_Context*        myContext;
    cudaStream_t        myStream;

    cudaSurfaceObject_t myInputSurface;
    cudaSurfaceObject_t myOutputSurface;

    pixelsort::PixelSorter  mySorter;

    const char*         myError;
};

#endif // PIXELSORT_TOP_H
