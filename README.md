# MujicaChk
The project hopes to realize efficient large model, large-scale training memory checkpoint and fast fault recovery

Part of the project code references dlrover-flashcheckpoint/Gemini

MujicaChk has no effect on the accuracy of the training and still recommends that users save normally with `torch.save` at the end of each epoch to ensure training.

# Get Started

MujicaChk builds on DeepSpeed and can be easily embedded into training

## Example

### Save

```
        >>> model, optimizer, _, lr_scheduler = deepspeed.initialize(...)
        >>> InmemoryCheckpointer = DeepSpeedCheckpointer(engine, save_dir) 
        >>> if args.save_model_step is not None and global_step % args.save_model_step == 0:
        >>>     InmemoryCheckpointer.save_checkpoint(save_dir)
```

### Load

```
        >>> model, optimizer, _, lr_scheduler = deepspeed.initialize(...)
        >>> InmemoryCheckpointer = DeepSpeedCheckpointer(engine, save_dir)
        >>> InmemoryCheckpointer.load_checkpoint(save_dir)
```
