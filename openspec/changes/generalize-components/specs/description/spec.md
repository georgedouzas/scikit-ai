# Spec Delta

## REMOVED Requirements

### Requirement: ImageDescriptor
**Reason**: Tasks are text generation named by input and output.
**Migration**: Use `ImageToTextGenerator(instructions='Describe the image.')`.

### Requirement: Descriptor learns from reference descriptions
**Reason**: Learning from reference outputs is part of every text generator.
**Migration**: Fit `ImageToTextGenerator` with reference descriptions as `y`.

### Requirement: Descriptor can be fine-tuned
**Reason**: Fine-tuning is part of every text generator.
**Migration**: Use `ImageToTextGenerator(adaptation='finetune')`.

### Requirement: AudioDescriptor
**Reason**: Tasks are text generation named by input and output.
**Migration**: Use `AudioToTextGenerator(operation='instruct', instructions='Describe the audio.')`.

### Requirement: VideoDescriptor
**Reason**: Tasks are text generation named by input and output.
**Migration**: Use `VideoToTextGenerator(instructions='Describe the video.')`.
