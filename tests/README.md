# ClearView test suite

700+ pytest tests covering models, losses, data loading, metrics, the
training loop and all three CLI scripts. The whole suite runs on CPU in a
few minutes using small synthetic images, with no datasets or downloads
needed. Tests that need CUDA or an optional package (`lpips`, `piq`,
`scipy`, `scikit-image`) skip cleanly when it is missing.

## Running

```bash
pip install -e ".[dev,metrics]"

pytest tests/                                   # everything
pytest tests/test_trainer.py                    # one file
pytest tests/test_trainer.py::TestTrainerResume # one class
pytest tests/ --cov=clearview --cov-report=term-missing
```

`make test` and `make test-cov` wrap the first and last commands.

## Layout

| Area            | Files                                                                                                                                                                          |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Models          | `test_models.py` (UNet, Attention UNet, building blocks), `test_resnet_unet.py`, `test_nafnet.py`, `test_restormer.py`, `test_histoformer.py`                                  |
| Losses          | `test_losses.py`, `test_perceptual.py`, `test_edge.py`, `test_frequency.py`, `test_adversarial.py`                                                                             |
| Data            | `test_data.py` (pair/folder/Rain100 datasets, transforms), `test_new_datasets.py` (Rain13K, DDN-Data, DID-Data, SPA-Data, GT-RAIN, RainDrop, RainDS, NTIRE haze, MixedDataset) |
| Metrics & utils | `test_metrics.py`, `test_niqe.py`, `test_utils.py`, `test_checkpoint.py`, `test_visualization.py`                                                                              |
| Training        | `test_trainer.py` (EMA, accumulation, resume, interrupts), `test_ema.py`, `test_schedulers.py`                                                                                 |
| CLI scripts     | `test_train_cli.py`, `test_train_script.py`, `test_train_mix_config.py`, `test_evaluate_script.py`, `test_inference_script.py`                                                 |
| End to end      | `test_integration.py`                                                                                                                                                          |

Shared fixtures live in `conftest.py`: `device`, `sample_image_tensor`,
`sample_image_pair`, `sample_batch`, `sample_numpy_image`,
`sample_pil_image`, `temp_image_file`, `temp_dir`, `sample_dataset_dir`,
`model_config`, `training_config`.

## Writing tests

- Put new tests in `tests/test_<area>.py`, with a type-hinted function and
  a one-line docstring for each test.
- Keep inputs tiny (16 to 64 px) and runs deterministic.
- Every bug fix should come with a test that fails without the fix.
