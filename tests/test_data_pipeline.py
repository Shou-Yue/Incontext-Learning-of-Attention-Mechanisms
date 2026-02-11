import torch

from src.data.task_sampler import LinearRegressionTaskSampler, build_prompt_tokens


def test_sampler_shapes():
    sampler = LinearRegressionTaskSampler(task_dim=20, device="cpu")
    batch = sampler.sample_batch(
        batch_size=4,
        n_context=10,
        active_dim=8,
        noise_std=0.1,
        x_distribution="isotropic",
        skew_power=2.0,
        orthant_mode="none",
    )

    assert batch["x_context"].shape == (4, 10, 20)
    assert batch["y_context"].shape == (4, 10)
    assert batch["x_query"].shape == (4, 1, 20)
    assert batch["y_query"].shape == (4, 1)


def test_prompt_builder():
    sampler = LinearRegressionTaskSampler(task_dim=6, device="cpu")
    batch = sampler.sample_batch(
        batch_size=2,
        n_context=4,
        active_dim=6,
        noise_std=0.0,
        x_distribution="isotropic",
        skew_power=2.0,
        orthant_mode="none",
    )
    tokens = build_prompt_tokens(batch["x_context"], batch["y_context"], batch["x_query"])
    assert tokens.shape == (2, 9, 6)
    # y token stores scalar in first coordinate.
    assert torch.allclose(tokens[:, 1::2, 1:], torch.zeros_like(tokens[:, 1::2, 1:]))

