#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ufra/engine.h"
#include <memory>

#define kPluginName "UFRa"
#define kPluginGrouping "MetaGPT/FaceReaging"
#define kPluginDescription "Universal Face Re-Aging (UFRa) Plugin - AI-powered face aging and de-aging"
#define kPluginIdentifier "com.metagpt.ufra"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

// Parameter names
#define kParamTargetAge "targetAge"
#define kParamProcessingMode "processingMode"
#define kParamIdentityLock "identityLock"
#define kParamTemporalStability "temporalStability"
#define kParamTextureKeep "textureKeep"
#define kParamSkinClean "skinClean"
#define kParamGrayDensity "grayDensity"

using namespace OFX;

class UFRaPlugin : public ImageEffect {
public:
    UFRaPlugin(OfxImageEffectHandle handle);
    virtual ~UFRaPlugin();

    virtual void render(const RenderArguments &args) override;
    virtual bool isIdentity(const IsIdentityArguments &args, Clip *&identityClip, double &identityTime) override;

private:
    void setupEngine();
    ufra::AgeControls getAgeControls(double time);
    ufra::ProcessingMode getProcessingMode(double time);

    // Clips
    Clip *srcClip_;
    Clip *dstClip_;

    // Parameters
    DoubleParam *targetAge_;
    ChoiceParam *processingMode_;
    DoubleParam *identityLock_;
    DoubleParam *temporalStability_;
    DoubleParam *textureKeep_;
    DoubleParam *skinClean_;
    DoubleParam *grayDensity_;

    // UFRa Engine
    std::unique_ptr<ufra::Engine> engine_;
    bool engineInitialized_;
};

UFRaPlugin::UFRaPlugin(OfxImageEffectHandle handle)
    : ImageEffect(handle), engineInitialized_(false) {
    srcClip_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
    dstClip_ = fetchClip(kOfxImageEffectOutputClipName);

    // Fetch parameters
    targetAge_ = fetchDoubleParam(kParamTargetAge);
    processingMode_ = fetchChoiceParam(kParamProcessingMode);
    identityLock_ = fetchDoubleParam(kParamIdentityLock);
    temporalStability_ = fetchDoubleParam(kParamTemporalStability);
    textureKeep_ = fetchDoubleParam(kParamTextureKeep);
    skinClean_ = fetchDoubleParam(kParamSkinClean);
    grayDensity_ = fetchDoubleParam(kParamGrayDensity);

    setupEngine();
}

UFRaPlugin::~UFRaPlugin() {}

void UFRaPlugin::setupEngine() {
    try {
        engine_ = ufra::createEngine();
        ufra::ModelConfig config;
        config.backend = ufra::GPUBackend::CUDA;
        config.batch_size = 1;
        config.use_half_precision = true;
        config.max_resolution = 1024;
        
        if (engine_->initialize(config)) {
            std::string modelPath = "/usr/local/share/ufra/models";
            if (engine_->loadModels(modelPath)) {
                engineInitialized_ = true;
            }
        }
    } catch (const std::exception& e) {
        // Error handling
    }
}

void UFRaPlugin::render(const RenderArguments &args) {
    if (!engineInitialized_) {
        throwSuiteStatusException(kOfxStatFailed);
        return;
    }

    auto_ptr<Image> src(srcClip_->fetchImage(args.time));
    auto_ptr<Image> dst(dstClip_->fetchImage(args.time));

    if (!src.get() || !dst.get()) {
        throwSuiteStatusException(kOfxStatFailed);
        return;
    }

    // Convert and process (simplified version)
    cv::Mat srcMat, dstMat;
    // ... conversion code ...
    
    ufra::FrameContext context;
    context.frame_number = static_cast<int>(args.time);
    context.controls = getAgeControls(args.time);
    context.mode = getProcessingMode(args.time);

    ufra::ProcessingResult result = engine_->processFrame(context);
    // ... result handling ...
}

ufra::AgeControls UFRaPlugin::getAgeControls(double time) {
    ufra::AgeControls controls;
    controls.target_age = static_cast<float>(targetAge_->getValueAtTime(time));
    controls.identity_lock_strength = static_cast<float>(identityLock_->getValueAtTime(time));
    return controls;
}

ufra::ProcessingMode UFRaPlugin::getProcessingMode(double time) {
    int mode;
    processingMode_->getValueAtTime(time, mode);
    return static_cast<ufra::ProcessingMode>(mode);
}

bool UFRaPlugin::isIdentity(const IsIdentityArguments &args, Clip *&identityClip, double &identityTime) {
    return false;
}

// Plugin registration
void getPluginIDs(PluginFactoryArray &ids) {
    static UFRaPluginFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
    ids.push_back(&p);
}