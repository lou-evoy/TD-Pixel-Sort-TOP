/* Pixel Sort TOP — TD SDK glue.
 * mirrors the CudaTOP sample (Samples/CPlusPlus/CudaTOP).
 */
#include "PixelSortTOP.h"

#include <cassert>
#include <cstdio>
#include <algorithm>

// keep in sync with customOPInfo.major/minorVersion below
static const char* kVersion = "1.0.0";

extern "C"
{

DLLEXPORT void
FillTOPPluginInfo(TOP_PluginInfo* info)
{
    if (!info->setAPIVersion(TOPCPlusPlusAPIVersion))
        return;

    info->executeMode = TOP_ExecuteMode::CUDA;

    // opType: A-Z then lowercase/digits, unique across installed TOPs
    info->customOPInfo.opType->setString("Pixelsort");
    info->customOPInfo.opLabel->setString("Pixel Sort");
    info->customOPInfo.opIcon->setString("PXS");
    info->customOPInfo.authorName->setString("SAT");
    info->customOPInfo.authorEmail->setString("levoy@sat.qc.ca");

    // 1-in / 1-out filter
    info->customOPInfo.minInputs = 1;
    info->customOPInfo.maxInputs = 1;

    info->customOPInfo.majorVersion = 1;
    info->customOPInfo.minorVersion = 0;
}

DLLEXPORT TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
    return new PixelSortTOP(info, context);
}

DLLEXPORT void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context* context)
{
    delete (PixelSortTOP*)instance;
}

} // extern "C"

// recreate every cook, never cache: bypass/reactivate frees the cudaArray, leaving a
// cached handle stale (sticky cudaErrorInvalidResourceHandle)
static void
setupCudaSurface(cudaSurfaceObject_t* surface, cudaArray_t array)
{
    if (*surface)
    {
        cudaDestroySurfaceObject(*surface);
        *surface = 0;
    }
    cudaResourceDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = array;
    cudaCreateSurfaceObject(surface, &desc);
}

static bool
isSupported8BitRGBA(OP_PixelFormat f)
{
    return f == OP_PixelFormat::BGRA8Fixed || f == OP_PixelFormat::RGBA8Fixed;
}

PixelSortTOP::PixelSortTOP(const OP_NodeInfo* info, TOP_Context* context) :
    myNodeInfo(info), myContext(context), myStream(0),
    myInputSurface(0), myOutputSurface(0), myError(nullptr)
{
    cudaStreamCreate(&myStream);
}

PixelSortTOP::~PixelSortTOP()
{
    if (myInputSurface)  cudaDestroySurfaceObject(myInputSurface);
    if (myOutputSurface) cudaDestroySurfaceObject(myOutputSurface);
    if (myStream)        cudaStreamDestroy(myStream);
    // mySorter frees its buffers
}

void
PixelSortTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs* inputs, void*)
{
    // cook on input/param change
    ginfo->cookEveryFrame = false;
    ginfo->cookEveryFrameIfAsked = false;
    // Version read-only (runs even with no input, when execute early-returns)
    if (inputs) inputs->enablePar("Version", false);
}

void
PixelSortTOP::getInfoPopupString(OP_String* info, void*)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "Pixel Sort v%s", kVersion);
    info->setString(buf);
}

void
PixelSortTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void*)
{
    myError = nullptr;
    inputs->enablePar("Version", false);

    if (inputs->getNumInputs() < 1)
    {
        myError = "Connect a TOP to the input.";
        return;
    }

    const OP_TOPInput* topInput = inputs->getInputTOP(0);
    if (!topInput)
    {
        myError = "Input TOP is invalid.";
        return;
    }

    const OP_TextureDesc& inDesc = topInput->textureDesc;

    if (inDesc.texDim != OP_TexDim::e2D)
    {
        myError = "Only 2D textures are supported (no 3D / cube / 2D-array).";
        return;
    }
    if (!isSupported8BitRGBA(inDesc.pixelFormat))
    {
        myError = "Input must be 8-bit RGBA/BGRA (BGRA8Fixed or RGBA8Fixed). "
                  "Convert upstream with a TOP set to 8-bit fixed.";
        return;
    }
    if ((int)inDesc.width  > pixelsort::kMaxDim ||
        (int)inDesc.height > pixelsort::kMaxDim)
    {
        myError = "Resolution exceeds the 8192-pixel limit of the composite key.";
        return;
    }

    // output matches input
    TOP_CUDAOutputInfo info;
    info.textureDesc = inDesc;
    info.stream      = myStream;

    // all input/param queries BEFORE beginCUDAOperations()
    OP_CUDAAcquireInfo acquireInfo;
    acquireInfo.stream = myStream;
    const OP_CUDAArrayInfo* inputArrayInfo = topInput->getCUDAArray(acquireInfo, nullptr);

    const OP_CUDAArrayInfo* outputArrayInfo = output->createCUDAArray(info, nullptr);
    if (!outputArrayInfo)
    {
        myError = "Failed to create output CUDA array.";
        return;
    }

    // gather + clamp params
    pixelsort::Params p;
    p.width  = (int)inDesc.width;
    p.height = (int)inDesc.height;
    p.bgra   = (inDesc.pixelFormat == OP_PixelFormat::BGRA8Fixed);

    p.axis    = (pixelsort::Axis)   std::clamp(inputs->getParInt("Axis"),    0, 1);
    p.order   = (pixelsort::Order)  std::clamp(inputs->getParInt("Order"),   0, 1);
    p.sortKey = (pixelsort::Channel)std::clamp(inputs->getParInt("Sortkey"), 0, 7);
    p.mode    = (pixelsort::Mode)   std::clamp(inputs->getParInt("Sortmode"),0, 5);
    p.amount  = (float)std::clamp(inputs->getParDouble("Amount"), 0.0, 1.0);
    p.seed    = (float)inputs->getParDouble("Seed");
    p.bypass  = inputs->getParInt("Bypass") != 0;

    // begin CUDA: cudaArray pointers valid now
    if (!myContext->beginCUDAOperations(nullptr))
    {
        myError = "beginCUDAOperations() failed.";
        return;
    }

    setupCudaSurface(&myOutputSurface, outputArrayInfo->cudaArray);
    if (inputArrayInfo && inputArrayInfo->cudaArray)
    {
        setupCudaSurface(&myInputSurface, inputArrayInfo->cudaArray);
    }
    else if (myInputSurface)
    {
        cudaDestroySurfaceObject(myInputSurface);
        myInputSurface = 0;
    }

    // swallow benign sticky error from surface (re)creation so process() reports only this cook
    cudaGetLastError();

    const char* algoError = nullptr;
    mySorter.process(myInputSurface, myOutputSurface, p, myStream, &algoError);
    if (algoError)
        myError = algoError;

    myContext->endCUDAOperations(nullptr);
}

void
PixelSortTOP::getErrorString(OP_String* error, void*)
{
    error->setString(myError);
}

void
PixelSortTOP::setupParameters(OP_ParameterManager* manager, void*)
{
    const char* sortPage = "Sort";

    // Sort Key channels; order matches pixelsort::Channel
    const char* chanNames[]  = { "Luminance", "Hue", "Saturation", "Value",
                                 "Red", "Green", "Blue", "Alpha" };
    const char* chanLabels[] = { "Luminance", "Hue", "Saturation", "Value",
                                 "Red", "Green", "Blue", "Alpha" };

    // Bypass — API doesn't report native Bypass to plugin, so expose our own
    {
        OP_NumericParameter np("Bypass");
        np.label = "Bypass";
        np.page  = sortPage;
        np.defaultValues[0] = 0.0;
        OP_ParAppendResult res = manager->appendToggle(np);
        assert(res == OP_ParAppendResult::Success);
    }
    {
        OP_StringParameter sp("Axis");
        sp.label = "Sort Axis";
        sp.page  = sortPage;
        sp.defaultValue = "Horizontal";
        const char* names[]  = { "Horizontal", "Vertical" };
        const char* labels[] = { "Horizontal", "Vertical" };
        OP_ParAppendResult res = manager->appendMenu(sp, 2, names, labels);
        assert(res == OP_ParAppendResult::Success);
    }
    {
        OP_StringParameter sp("Order");
        sp.label = "Sort Order";
        sp.page  = sortPage;
        sp.defaultValue = "Ascending";
        const char* names[]  = { "Ascending", "Descending" };
        const char* labels[] = { "Ascending", "Descending" };
        OP_ParAppendResult res = manager->appendMenu(sp, 2, names, labels);
        assert(res == OP_ParAppendResult::Success);
    }
    {
        OP_StringParameter sp("Sortkey");
        sp.label = "Sort Key";
        sp.page  = sortPage;
        sp.defaultValue = "Luminance";
        OP_ParAppendResult res = manager->appendMenu(sp, 8, chanNames, chanLabels);
        assert(res == OP_ParAppendResult::Success);
    }
    // Sort Mode — interval/reveal style driven by Sort Amount
    {
        OP_StringParameter sp("Sortmode");
        sp.label = "Sort Mode";
        sp.page  = sortPage;
        // default to a mode where Amount is immediately live
        sp.defaultValue = "Revealdark";
        const char* names[]  = { "Full", "Revealdark", "Revealbright",
                                 "Random", "Edges", "Melt" };
        const char* labels[] = { "Full", "Reveal Dark", "Reveal Bright",
                                 "Random Intervals", "Edges", "Melt" };
        OP_ParAppendResult res = manager->appendMenu(sp, 6, names, labels);
        assert(res == OP_ParAppendResult::Success);
    }
    // Sort Amount — drives the selected Sort Mode (0..1)
    {
        OP_NumericParameter np("Amount");
        np.label = "Sort Amount";
        np.page  = sortPage;
        np.defaultValues[0] = 0.5;
        np.minValues[0] = 0.0;  np.maxValues[0] = 1.0;
        np.minSliders[0] = 0.0; np.maxSliders[0] = 1.0;
        np.clampMins[0] = true; np.clampMaxes[0] = true;
        OP_ParAppendResult res = manager->appendFloat(np);
        assert(res == OP_ParAppendResult::Success);
    }
    // Seed — Random Intervals pattern (animate for evolving randomness)
    {
        OP_NumericParameter np("Seed");
        np.label = "Seed";
        np.page  = sortPage;
        np.defaultValues[0] = 0.0;
        np.minValues[0] = 0.0;  np.maxValues[0] = 100.0;
        np.minSliders[0] = 0.0; np.maxSliders[0] = 100.0;
        np.clampMins[0] = true; // clamp low end only
        OP_ParAppendResult res = manager->appendFloat(np);
        assert(res == OP_ParAppendResult::Success);
    }
    // Version — read-only, shows the build version
    {
        OP_StringParameter sp("Version");
        sp.label = "Version";
        sp.page  = "Version";
        sp.defaultValue = kVersion;
        OP_ParAppendResult res = manager->appendString(sp);
        assert(res == OP_ParAppendResult::Success);
    }
}
