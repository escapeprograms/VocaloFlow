# AdversarialPostnet — Memory Palace

Experiment: a lightweight post-network trained with adversarial + reconstruction losses to sharpen VocaloFlow's over-smoothed mel-spectrogram output. The post-network takes VocaloFlow's 32-step ODE output and produces a sharpened mel, supervised by a patch discriminator (WeSinger 2 style random area crops).

## Module Structure

```
AdversarialPostnet/
    configs/postnet_config.py           -- PostnetConfig dataclass
    data/generate_predictions.py        -- CLI: frozen VocaloFlow inference -> predicted_mel.npy
    data/postnet_dataset.py             -- PostnetDataset for (predicted_mel, target_mel) pairs
    data/collate.py                     -- postnet_collate_fn
    model/postnet.py                    -- PostNet: residual Conv1d refinement network
    model/discriminator.py              -- PatchDiscriminator: 2D conv patch classifier
    training/losses.py                  -- hinge losses, feature matching, masked L1
    training/random_crop.py             -- Multi-scale random area cropping
    training/checkpoint.py              -- Save/load for two models + two optimizers
    training/train_postnet.py           -- Main training loop (entry point)
```

## configs/postnet_config.py

`PostnetConfig` dataclass with YAML serialization (same pattern as `VocaloFlowConfig`).

- **Generation fields**: `data_dir`, `manifest_path`, `vocaloflow_checkpoint` (defaults to 4-16-wavenet), `max_dtw_cost`, `max_songs` (0=all), `song_subset_seed`
- **Training data fields**: `predicted_mel_manifest`, `max_seq_len` (256), `val_fraction` (0.05), `seed` (42), `split_mode` ("song")
- **Inference fields**: `num_ode_steps` (32), `ode_method` ("midpoint"), `cfg_scale` (2.0), `generation_batch_size` (16)
- **PostNet architecture**: `postnet_num_blocks` (4), `postnet_kernel_size` (3)
- **Discriminator architecture**: `disc_channels` ([32, 64, 128, 256])
- **Loss weights**: `lambda_adv` (0.1), `lambda_fm` (2.0)
- **Training**: `learning_rate` (2e-4), `disc_learning_rate` (2e-4), `adam_beta1` (0.8), `adam_beta2` (0.99), `weight_decay` (0.01), `grad_clip` (1.0), `total_steps` (20000), `warmup_steps` (0), `disc_warmup_steps` (2000), `adv_ramp_steps` (1000)
- **Logging**: `log_every` (50), `val_every` (1000), `save_every` (5000)
- **Crop specs**: `crop_specs` ([[32,64], [64,128], [128,128]])
- **to_yaml(path)** / **from_yaml(path)**: YAML serialization

## data/generate_predictions.py

CLI entry point for generating predicted mels from a frozen VocaloFlow model.

- **subset_songs(df, max_songs, seed)**: deterministically selects `max_songs` dali_ids using `np.random.RandomState(seed)`
- **_load_chunk(row, max_seq_len)**: loads chunk inputs from disk, resolves phoneme indirection, pads/truncates with deterministic start=0
- **build_batch(records, max_seq_len, device)**: loads multiple chunks, stacks into batched tensors, builds padding_mask
- **generate(args)**: main loop — loads manifest, subsets, resumes, runs batched `sample_ode`, saves `predicted_mel.npy`, writes output manifest

## data/postnet_dataset.py

`PostnetDataset(Dataset)` — loads `(predicted_mel, target_mel)` pairs. Random crop if training, start=0 if val. Returns: `predicted_mel (T,128)`, `target_mel (T,128)`, `padding_mask (T,)`, `length (int)`.

## data/collate.py

`postnet_collate_fn(batch)` — stacks items into: `predicted_mel (B,T,128)`, `target_mel (B,T,128)`, `padding_mask (B,T)`, `length (B,)`.

## model/postnet.py

~260K parameters. Residual Conv1d stack operating on mel spectrograms.

- **_kaiming_init_conv(conv)**: Kaiming-normal init with zero bias (local copy of VocaloFlow convention)
- **ResidualConv1dBlock(channels, kernel_size=3)**: Conv1d -> LeakyReLU(0.2) -> Conv1d + residual skip. Operates in (B, C, T) channels-first format. Kaiming init on both convs.
- **PostNet(mel_channels=128, num_blocks=4, kernel_size=3)**: nn.ModuleList of ResidualConv1dBlocks + final 1x1 Conv1d. Forward: (B, T, 128) -> transpose -> blocks -> 1x1 conv -> transpose -> add input (global residual) -> (B, T, 128). The global residual means the network starts as near-identity.

## model/discriminator.py

~1.2M parameters. 2D patch discriminator for mel-spectrogram crops.

- **PatchDiscriminator(channels=[32, 64, 128, 256])**: 4 downsampling Conv2d layers (kernel=4, stride=2, padding=1) with weight normalization (torch.nn.utils.parametrizations.weight_norm) and LeakyReLU(0.2), followed by 1 output Conv2d (kernel=3, stride=1, padding=1) producing raw logits. No weight norm on the output layer.
- **forward(x: (B,1,T,F))**: returns `(logits, features)` where `logits` is (B,1,H',W') spatial score map and `features` is list of 4 post-LeakyReLU intermediate activations for feature matching loss.

## training/losses.py

Plain functions (not nn.Module) for all loss computations:

- **masked_l1(pred, target, padding_mask)**: L1 loss masked by (B,T) bool mask, averaged over valid frames and mel bins
- **hinge_d_loss(real_scores, fake_scores)**: `mean(relu(1-real)) + mean(relu(1+fake))`
- **hinge_g_loss(fake_scores)**: `-mean(fake_scores)`
- **feature_matching_loss(real_features, fake_features)**: L1 distance between detached real features and fake features, summed across layers, divided by layer count

## training/random_crop.py

Multi-scale random area cropping (WeSinger 2 style):

- **CropSpec(time_frames, mel_bins)**: dataclass specifying one crop scale
- **extract_random_crops(real_mel, fake_mel, crop_specs, padding_mask)**: for each spec, samples random (t_start, f_start) per batch item (same coords for real and fake), pads if mel shorter than crop, returns list of (real_crop, fake_crop) tuples each shaped (B, 1, T_crop, F_crop)

## training/checkpoint.py

Same pattern as VocaloFlow's checkpoint.py but for two models + two optimizers:

- **save_checkpoint(postnet, discriminator, opt_g, opt_d, step, config, wandb_run_id)**: saves all state dicts + config + step
- **load_checkpoint(path, device)**: torch.load with weights_only=False
- **find_latest_checkpoint(checkpoint_dir)**: glob for checkpoint_*.pt, return highest step

## training/train_postnet.py

Main training entry point. CLI: `python -m training.train_postnet --name <run> --predicted-mel-manifest <path>`

- **get_effective_lambda_adv(step, config)**: returns 0 during disc_warmup_steps, then linearly ramps to lambda_adv over adv_ramp_steps
- **validate(postnet, val_loader, device)**: @torch.no_grad(), computes masked L1 on val set
- **train(config)**: main loop following VocaloFlow/training/train.py pattern:
  1. Resume detection from latest checkpoint
  2. wandb + TensorBoard logging setup
  3. Data: load predicted_mel_manifest, split by song, create DataLoaders
  4. Models: PostNet + PatchDiscriminator
  5. Optimizers: separate AdamW for G and D with betas=(0.8, 0.99)
  6. Per-step: postnet forward -> random crops -> D update (detached fakes, hinge loss) -> G update (L_rec + λ_adv * L_G_adv + λ_fm * L_fm)
  7. Logs: train/loss_total, train/loss_rec, train/loss_g_adv, train/loss_fm, train/loss_d, train/lambda_adv, train/d_real_mean, train/d_fake_mean, val/loss_rec
- **_load_config_from_checkpoint(run_name)**: reconstructs PostnetConfig from saved checkpoint
- **_apply_yaml_overrides(config, path)**: merges YAML overrides onto config
- **main()**: argparse CLI with --name, --resume, --config, --predicted-mel-manifest, --batch-size, --lr, --total-steps




python -m training.train_postnet 
    --name my-run 
    --predicted-mel-manifest predicted_mel_manifest.csv


python evaluate_postnet.py 
    --input-mel ../demo/let_it_go/4-16-wavenet/output_mel.npy 
    --checkpoint checkpoints/my-run/checkpoint_20000.pt 
    --output-dir eval_output/let_it_go


python evaluate_postnet.py 
    --input-mel ../demo/we_are_charlie/4-16-wavenet/output_mel.npy 
    --checkpoint checkpoints/my-run/checkpoint_20000.pt 
    --output-dir eval_output/we_are_charlie