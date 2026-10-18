# Spec Delta

## REMOVED Requirements

### Requirement: ImageDescriptor
**Reason**: Descriptions are text generation with instructions.
**Migration**: Use `TextGenerator(input='image', instructions='Describe the image.')`, or `skai.presets.describer('image')`.

### Requirement: Descriptor learns from reference descriptions
**Reason**: Learning from references is part of `TextGenerator` for every input type.
**Migration**: Fit `TextGenerator(input='image', ...)` with reference descriptions as `y`.

### Requirement: Descriptor can be fine-tuned
**Reason**: Fine-tuning is part of `TextGenerator`.
**Migration**: Use `TextGenerator(input='image', adaptation='finetune', ...)`.

### Requirement: AudioDescriptor
**Reason**: Descriptions are text generation with instructions.
**Migration**: Use `TextGenerator(input='audio', instructions='Describe the audio.')`, or `skai.presets.describer('audio')`.

### Requirement: VideoDescriptor
**Reason**: Descriptions are text generation with instructions.
**Migration**: Use `TextGenerator(input='video', instructions='Describe the video.', n_frames=...)`, or `skai.presets.describer('video')`.
