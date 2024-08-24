# MujicaChk
The project hopes to realize efficient large model, large-scale training memory checkpoint and fast fault recovery

Part of the project code references dlrover-flashcheckpoint/Gemini

#Get Started

MujicaChk builds on DeepSpeed and can be easily embedded into training

##Example

###Save
'''
        >>> model, optimizer, _, lr_scheduler = deepspeed.initialize(...)
        >>> MujicaCheckpointer = DeepSpeedCheckpointer(engine, save_dir) 
        >>> if args.save_model_step is not None and global_step % args.save_model_step == 0:
        >>>     MujicaCheckpointer.save_checkpoint( save_dir)
'''

###Load
'''
        >>> model, optimizer, _, lr_scheduler = deepspeed.initialize(...)
        >>> MujicaCheckpointer = DeepSpeedCheckpointer(engine, save_dir)
        >>> MujicaCheckpointer.load_checkpoint( save_dir)
'''
