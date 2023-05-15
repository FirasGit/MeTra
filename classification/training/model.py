from classification.models import MultiModalPretrainedViTLab


def get_model(cfg):
    if cfg.model.name == 'MultiModalPretrainedViTLab':
        return MultiModalPretrainedViTLab(
            	model_name=cfg.model.type,
                image_size=cfg.model.transforms.img_size, 
                num_classes=cfg.model.output_logits, 
                channels=cfg.model.num_channels, 
                **cfg.model.meta)

    raise KeyError(f'Model {cfg.model.name} is not supported')
