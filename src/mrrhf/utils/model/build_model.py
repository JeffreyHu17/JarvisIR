from .base_model.modeling_llava import LlavaForConditionalGeneration
from transformers import AutoProcessor

def build_model(args=None,
                model_architecture=None,
                from_checkpoint=None):
    
    if model_architecture is None:
        model_architecture = args.model_architecture

    if from_checkpoint is None:
        from_checkpoint = args.from_checkpoint
    
    if model_architecture=="llava":
        model = LlavaForConditionalGeneration.from_pretrained(
                from_checkpoint, 
                low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(from_checkpoint)

        image_processor = processor.image_processor
        tokenizer = processor.tokenizer
        tokenizer.padding_side = 'right'
        
        # freeze parameters
        model.vision_tower.requires_grad_(False)
        model.multi_modal_projector.requires_grad_(True)

        if args.lang_decoder_update:
            model.language_model.requires_grad_(True)
        else:
            model.language_model.requires_grad_(False)

        return model, image_processor, tokenizer
    else:
        assert "Please add this model architeacture in build_model.py!"
