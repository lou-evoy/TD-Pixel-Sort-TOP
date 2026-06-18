/* Pixel Sort TOP — TouchDesigner SDK glue.
 *
 * Mirrors the structure of the shipped CudaTOP sample (Samples/CPlusPlus/CudaTOP):
 *   - executeMode = TOP_ExecuteMode::CUDA
 *   - acquire input texture as a cudaArray via OP_TOPInput::getCUDAArray()
 *   - create the output cudaArray via TOP_Output::createCUDAArray()
 *   - wrap both as cudaSurfaceObjects, do work between begin/endCUDAOperations()
 *   - use one instance-owned cudaStream_t for the whole node's lifetime
 */
#include "PixelSortTOP.h"

#include <cassert>
#include <cstdio>
#include <algorithm>

// Single source of truth for the plugin version. Bump on each release; keep the numeric
// parts in sync with customOPInfo.major/minorVersion below.
static const char* kVersion = "1.0.0";

// ----------------------------- DLL entry points -----------------------------
extern "C"
{

DLLEXPORT void
FillTOPPluginInfo(TOP_PluginInfo* info)
{
    if (!info->setAPIVersion(TOPCPlusPlusAPIVersion))
        return;

    info->executeMode = TOP_ExecuteMode::CUDA;

    // opType: must start with A-Z then lowercase/digits, unique across installed TOPs.
    info->customOPInfo.opType->setString("Pixelsort");
    info->customOPInfo.opLabel->setString("Pixel Sort");
    info->customOPInfo.opIcon->setString("PXS");
    info->customOPInfo.authorName->setString("SAT");
    info->customOPInfo.authorEmail->setString("levoy@sat.qc.ca");

    // Pixel sorting is a 1-in / 1-out filter.
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

// --------------------------------- helpers ----------------------------------
// Bind a fresh surface object to 'array'. We destroy any previous object and create
// a new one every cook rather than caching it. Caching (as the CudaTOP sample does,
// keyed on cudaGetSurfaceObjectResourceDesc) is fragile: when the node is bypassed and
// reactivated, TD frees the old cudaArray and re-registers its interop, leaving the
// cached surface handle stale. Querying a stale handle returns cudaErrorInvalidResource
// Handle (400) as a *sticky* error that then surfaces at the next kernel-launch check.
// Surface creation is only a few microseconds, so recreating each cook is the robust choice.
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

// ------------------------------- PixelSortTOP --------------------------------
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
    // mySorter frees its device buffers in its own destructor.
}

void
PixelSortTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs* inputs, void*)
{
    // Cook when the input or a parameter changes (default filter behavior).
    ginfo->cookEveryFrame = false;
    ginfo->cookEveryFrameIfAsked = false;
    // Keep the Version field read-only here too (runs even when no input is connected
    // and execute would early-return).
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
    inputs->enablePar("Version", false);   // make the Version field read-only

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

    // Output matches the input texture (size / format / dim).
    TOP_CUDAOutputInfo info;
    info.textureDesc = inDesc;
    info.stream      = myStream;

    // --- All input/parameter queries must happen BEFORE beginCUDAOperations() ---
    OP_CUDAAcquireInfo acquireInfo;
    acquireInfo.stream = myStream;
    const OP_CUDAArrayInfo* inputArrayInfo = topInput->getCUDAArray(acquireInfo, nullptr);

    const OP_CUDAArrayInfo* outputArrayInfo = output->createCUDAArray(info, nullptr);
    if (!outputArrayInfo)
    {
        myError = "Failed to create output CUDA array.";
        return;
    }

    // Gather parameters and clamp to valid ranges.
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

    // --- Begin CUDA: cudaArray pointers become valid now ---
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

    // Swallow any benign sticky error from (re)creating surface objects above (e.g.
    // after a bypass toggle), so the launch checks inside process() only report errors
    // from this cook's own kernels.
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

// ------------------------------- parameters ----------------------------------
void
PixelSortTOP::setupParameters(OP_ParameterManager* manager, void*)
{
    const char* sortPage = "Sort";

    // The 8-channel set for Sort Key. 'names' are the stored values; their order
    // matches pixelsort::Channel.
    const char* chanNames[]  = { "Luminance", "Hue", "Saturation", "Value",
                                 "Red", "Green", "Blue", "Alpha" };
    const char* chanLabels[] = { "Luminance", "Hue", "Saturation", "Value",
                                 "Red", "Green", "Blue", "Alpha" };

    // Bypass — pass the input through untouched (the C++ TOP API does not report the
    // node's native Bypass flag to the plugin, so we expose our own that actually works).
    {
        OP_NumericParameter np("Bypass");
        np.label = "Bypass";
        np.page  = sortPage;
        np.defaultValues[0] = 0.0;
        OP_ParAppendResult res = manager->appendToggle(np);
        assert(res == OP_ParAppendResult::Success);
    }
    // Sort Axis
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
    // Sort Order
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
    // Sort Key
    {
        OP_StringParameter sp("Sortkey");
        sp.label = "Sort Key";
        sp.page  = sortPage;
        sp.defaultValue = "Luminance";
        OP_ParAppendResult res = manager->appendMenu(sp, 8, chanNames, chanLabels);
        assert(res == OP_ParAppendResult::Success);
    }
    // Sort Mode — the interval/reveal style driven by Sort Amount.
    {
        OP_StringParameter sp("Sortmode");
        sp.label = "Sort Mode";
        sp.page  = sortPage;
        // Default to a mode where the Amount slider is immediately live (at 1.0 it
        // equals Full; lowering it reveals progressively).
        sp.defaultValue = "Revealdark";
        const char* names[]  = { "Full", "Revealdark", "Revealbright",
                                 "Random", "Edges", "Melt" };
        const char* labels[] = { "Full", "Reveal Dark", "Reveal Bright",
                                 "Random Intervals", "Edges", "Melt" };
        OP_ParAppendResult res = manager->appendMenu(sp, 6, names, labels);
        assert(res == OP_ParAppendResult::Success);
    }
    // Sort Amount — single slider that drives the selected Sort Mode (0..1).
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
    // Seed — random pattern for Random Intervals mode (animate for evolving randomness).
    {
        OP_NumericParameter np("Seed");
        np.label = "Seed";
        np.page  = sortPage;
        np.defaultValues[0] = 0.0;
        np.minValues[0] = 0.0;  np.maxValues[0] = 100.0;
        np.minSliders[0] = 0.0; np.maxSliders[0] = 100.0;
        np.clampMins[0] = true; // allow large values; only clamp the low end
        OP_ParAppendResult res = manager->appendFloat(np);
        assert(res == OP_ParAppendResult::Success);
    }
    // Version — read-only string on its own tab (shows the version the node was made with).
    {
        OP_StringParameter sp("Version");
        sp.label = "Version";
        sp.page  = "Version";
        sp.defaultValue = kVersion;
        OP_ParAppendResult res = manager->appendString(sp);
        assert(res == OP_ParAppendResult::Success);
    }
}
