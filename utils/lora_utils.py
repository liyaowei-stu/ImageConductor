from peft import LoraConfig

def add_LoRA_to_controlnet(lora_rank, controlnet):
    controlnet_down_blocks = [f"controlnet_down_blocks.{i}" for i in range(12)]
    target_modules = ['to_q', 'to_k', 'to_v','to_out.0', 'conv', 'proj',
                        'proj_in', 'proj_out', 'time_emb_proj', 
                        'linear_1', 'linear_2', 'ff.net.2', 'conv_shortcut', 
                        'controlnet_cond_embedding', 
                        'conv_in', 'conv1', 'conv2',
                        'controlnet_mid_block',
                        ]
    target_modules = target_modules + controlnet_down_blocks

    # object motion control module
    omcm_lora_config = LoraConfig(
        r=lora_rank,
        use_dora=False,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    # camera motion control module
    cmcm_lora_config = LoraConfig(
        r=lora_rank,
        use_dora=False,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    controlnet.add_adapter(omcm_lora_config, "omcm_weights")
    controlnet.add_adapter(cmcm_lora_config, "cmcm_weights")